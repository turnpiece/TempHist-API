"""OG image endpoint — GET /v1/og/{share_id}.png

Returns a 1200×630 PNG chart image for use as an Open Graph preview.
No authentication required (social media crawlers don't have app credentials).
"""
import io
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import redis
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from cache.keys import bundle_key, normalize_location_for_cache, rec_key
from routers.dependencies import get_redis_client
from utils.share_store import get_share_store

logger = logging.getLogger(__name__)
router = APIRouter()

_SHARE_CACHE_TTL = 30 * 24 * 3600  # 30 days — share records never change
_IMG_W, _IMG_H = 1200, 630

# Brand colours — match the website / mobile app
_BG = "#242456"
_REF_YEAR = "#51cf66"
_AVG_LINE = "#ffffff"
_TICK_COLOR = "#cccccc"
_TITLE_COLOR = (1.0, 1.0, 1.0, 0x8C / 0xFF)  # white at ~55% opacity
_FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]

# Climate-stripes palette — matches web/src/constants/index.ts
_NEUTRAL_RGB = (0x8E, 0x8E, 0x93)
_WARM_RGB    = (0xFF, 0x3B, 0x30)
_COOL_RGB    = (0x3B, 0x82, 0xF6)
_BAR_NEUTRAL_Z    = 0.25
_BAR_SATURATION_Z = 2.0


def _share_cache_key(share_id: str) -> str:
    return f"share:{share_id}"


_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _format_period_heading(period: str, identifier: str) -> str:
    """Return a human heading like 'Month ending 7th April'."""
    parts = identifier.split("-") if identifier else []
    if len(parts) != 2:
        return identifier or ""
    try:
        month = int(parts[0])
        day = int(parts[1])
    except ValueError:
        return identifier
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return identifier
    friendly = f"{_ordinal(day)} {_MONTH_NAMES[month - 1]}"
    p = (period or "").lower()
    if p == "daily":
        return friendly
    if p == "weekly":
        return f"Week ending {friendly}"
    if p == "monthly":
        return f"Month ending {friendly}"
    if p == "yearly":
        return f"Year ending {friendly}"
    return friendly


async def _lookup_share(share_id: str, redis_client: redis.Redis) -> Optional[dict]:
    """Return share record from Redis cache or Postgres, or None if not found."""
    cache_key = _share_cache_key(share_id)
    try:
        cached = redis_client.get(cache_key)
        if cached:
            data_str = cached.decode("utf-8") if isinstance(cached, bytes) else cached
            return json.loads(data_str)
    except Exception as exc:
        logger.warning("Redis read failed for share %s: %s", share_id, exc)

    store = get_share_store()
    share = await store.get_share(share_id)
    if share is None:
        return None

    try:
        redis_client.setex(cache_key, _SHARE_CACHE_TTL, json.dumps(share))
    except Exception as exc:
        logger.warning("Redis write failed for share %s: %s", share_id, exc)

    return share


def _get_bundle_records(share: dict, redis_client: redis.Redis) -> Optional[list]:
    """Return per-year temperature records from Redis, or None.

    Tries the assembled bundle first (written when the GET records endpoint is
    called).  Falls back to reading per-year records directly (written by the
    async job), so share pages that only use the async flow still get a chart.
    """
    slug = normalize_location_for_cache(share["location"])
    period = share["period"]
    identifier = share["identifier"]

    # 1. Try the bundle key first — fastest path
    bkey = bundle_key(period, slug, identifier)
    try:
        raw = redis_client.get(bkey)
        if raw:
            data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            records = data.get("records", [])
            if records:
                return records
    except Exception as exc:
        logger.warning("Redis bundle read failed for share %s: %s", share.get("id"), exc)

    # 2. Bundle absent — assemble from per-year records written by the async job
    logger.info("OG bundle miss for share=%s, falling back to per-year records", share.get("id"))
    current_year = datetime.now(timezone.utc).year
    years = list(range(1950, current_year + 1))
    keys = [rec_key(period, slug, identifier, y) for y in years]
    try:
        values = redis_client.mget(keys)
        records = []
        for year, value in zip(years, values):
            if value:
                try:
                    records.append(json.loads(value.decode() if isinstance(value, bytes) else value))
                except Exception:
                    pass
        if records:
            logger.info("OG per-year fallback: share=%s found %d records", share.get("id"), len(records))
            return records
    except Exception as exc:
        logger.warning("Redis per-year read failed for share %s: %s", share.get("id"), exc)

    return None


def _celsius_to_fahrenheit(t: float) -> float:
    return t * 9 / 5 + 32


def _lerp_color(
    a: tuple,
    b: tuple,
    t: float,
) -> tuple:
    """Interpolate between two (0-255) RGB tuples; return matplotlib (0-1) tuple."""
    return (
        round(a[0] + (b[0] - a[0]) * t) / 255,
        round(a[1] + (b[1] - a[1]) * t) / 255,
        round(a[2] + (b[2] - a[2]) * t) / 255,
    )


def _bar_color_for_z_score(z: float) -> tuple:
    """Map a Z-score to a cool/neutral/warm colour matching the frontend logic."""
    magnitude = abs(z)
    if magnitude <= _BAR_NEUTRAL_Z:
        return tuple(c / 255 for c in _NEUTRAL_RGB)
    blend = min(1.0, (magnitude - _BAR_NEUTRAL_Z) / (_BAR_SATURATION_Z - _BAR_NEUTRAL_Z))
    return _lerp_color(_NEUTRAL_RGB, _WARM_RGB if z >= 0 else _COOL_RGB, blend)


def _compute_bar_colors(years: list, temps: list, ref_year) -> list:
    """Return one matplotlib color per bar using the climate-stripes Z-score scheme."""
    hist_temps = [t for y, t in zip(years, temps) if y != ref_year]
    if not hist_temps:
        return [_REF_YEAR if y == ref_year else tuple(c / 255 for c in _NEUTRAL_RGB)
                for y in years]
    mean = sum(hist_temps) / len(hist_temps)
    # Population std dev — matches calculate_standard_deviation() in utils/temperature.py
    variance = sum((t - mean) ** 2 for t in hist_temps) / len(hist_temps)
    std_dev = variance ** 0.5
    colors = []
    for y, t in zip(years, temps):
        if y == ref_year:
            colors.append(_REF_YEAR)
        elif std_dev > 0:
            colors.append(_bar_color_for_z_score((t - mean) / std_dev))
        else:
            colors.append(tuple(c / 255 for c in _NEUTRAL_RGB))
    return colors


def _render_chart(share: dict, records: list) -> bytes:
    """Render a horizontal bar chart of per-year temperatures as PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FuncFormatter

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = _FONT_FAMILY

    unit = share.get("unit", "celsius")
    ref_year = share.get("ref_year")
    location = share.get("location", "")
    period = share.get("period", "")
    identifier = share.get("identifier", "")
    unit_symbol = "°C" if unit == "celsius" else "°F"

    # Build parallel lists, filtering out null temperatures
    pairs = [
        (r["year"], r["temperature"])
        for r in records
        if r.get("temperature") is not None and r.get("year") is not None
    ]
    if not pairs:
        return _render_placeholder(share)

    # Sort ascending by year, then reverse so most recent is at the top
    pairs.sort(key=lambda p: p[0])
    years, temps = zip(*pairs)
    years = list(reversed(years))
    temps = list(reversed(temps))

    if unit == "fahrenheit":
        temps = [_celsius_to_fahrenheit(t) for t in temps]

    fig = plt.figure(figsize=(_IMG_W / 100, _IMG_H / 100), dpi=100)
    fig.patch.set_facecolor(_BG)
    ax = fig.add_axes([0.24, 0.06, 0.52, 0.80])
    ax.set_facecolor(_BG)

    # Historical average (excludes ref_year) — white dotted vertical line
    hist_temps = [t for y, t in zip(years, temps) if y != ref_year]
    if hist_temps:
        avg = sum(hist_temps) / len(hist_temps)
        ax.axvline(
            avg,
            color=_AVG_LINE,
            linestyle=":",
            linewidth=2,
            alpha=0.9,
            zorder=1,
        )

    # Horizontal bars: ref_year in green, historical years in climate-stripes colours
    bar_colors = _compute_bar_colors(years, temps, ref_year)
    ax.barh(years, temps, color=bar_colors, height=0.75, zorder=2, alpha=0.9)

    # Annotate ref_year bar with its value
    if ref_year in years:
        idx = years.index(ref_year)
        ref_temp = temps[idx]
        ax.annotate(
            f"{ref_temp:.0f}{unit_symbol}" if unit == "fahrenheit" else f"{ref_temp:.1f}{unit_symbol}",
            xy=(ref_temp, ref_year),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=_REF_YEAR,
            fontsize=13,
            fontweight="bold",
        )

    # X-axis limits: start a few degrees below the lowest temperature
    # (not at 0 — temperatures may be negative)
    t_min = min(temps)
    t_max = max(temps)
    span = t_max - t_min
    pad_low = max(span * 0.15, 2.0)
    pad_high = max(span * 0.12, 1.5)
    ax.set_xlim(t_min - pad_low, t_max + pad_high)

    # Title — city name + period heading
    city = location.split(",")[0].strip() if location else ""
    heading = _format_period_heading(period, identifier)
    title = f"{city} · {heading}" if city and heading else (city or heading)
    ax.set_title(
        title,
        loc="left",
        color=_TITLE_COLOR,
        fontsize=22,
        pad=12,
        fontweight="normal",
        fontfamily=_FONT_FAMILY,
    )

    ax.xaxis.tick_top()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}{unit_symbol}"))
    ax.tick_params(colors=_TICK_COLOR, which="both", labelsize=11)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    for spine in ax.spines.values():
        spine.set_edgecolor(_BG)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_placeholder(share: dict) -> bytes:
    """Render a simple branded placeholder when chart data is unavailable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = _FONT_FAMILY

    location = share.get("location", "")
    city = location.split(",")[0].strip() if location else ""
    heading = _format_period_heading(share.get("period", ""), share.get("identifier", ""))

    fig = plt.figure(figsize=(_IMG_W / 100, _IMG_H / 100), dpi=100)
    fig.patch.set_facecolor(_BG)
    ax = fig.add_axes([0.24, 0.06, 0.52, 0.80])
    ax.set_facecolor(_BG)
    ax.axis("off")

    ax.text(
        0.5, 0.58, "TempHist",
        ha="center", va="center", transform=ax.transAxes,
        color="white", fontsize=64, fontweight="normal",
        fontfamily=_FONT_FAMILY,
    )
    if city:
        ax.text(
            0.5, 0.46, city,
            ha="center", va="center", transform=ax.transAxes,
            color=_TITLE_COLOR, fontsize=30, fontweight="normal",
            fontfamily=_FONT_FAMILY,
        )
    if heading:
        ax.text(
            0.5, 0.36, heading,
            ha="center", va="center", transform=ax.transAxes,
            color=_TICK_COLOR, fontsize=20,
            fontfamily=_FONT_FAMILY,
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@router.get("/v1/og/{share_id}.png", include_in_schema=True)
async def og_image(
    share_id: str,
    redis_client: redis.Redis = Depends(get_redis_client),
):
    """Return a 1200×630 OG preview image for the given share ID. No auth required."""
    if len(share_id) != 8 or not share_id.isalnum():
        raise HTTPException(status_code=404, detail="Share not found.")

    share = await _lookup_share(share_id, redis_client)
    if share is None:
        raise HTTPException(status_code=404, detail="Share not found.")

    records = _get_bundle_records(share, redis_client)

    try:
        png = _render_chart(share, records) if records else _render_placeholder(share)
    except Exception as exc:
        logger.error("OG image render failed for share %s: %s", share_id, exc, exc_info=True)
        try:
            png = _render_placeholder(share)
        except Exception:
            raise HTTPException(status_code=500, detail="Image generation failed.")

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )
