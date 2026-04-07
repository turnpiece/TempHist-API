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

from cache_utils import bundle_key, normalize_location_for_cache, rec_key
from routers.dependencies import get_redis_client
from utils.share_store import get_share_store

logger = logging.getLogger(__name__)
router = APIRouter()

_SHARE_CACHE_TTL = 30 * 24 * 3600  # 30 days — share records never change
_IMG_W, _IMG_H = 1200, 630

# Dark-themed brand colours
_BG_DARK = "#1a1a2e"
_BG_AXES = "#16213e"
_BAR = "#ff6b6b"
_REF_YEAR = "#51cf66"
_AVG_LINE = "#4dabf7"
_TICK_COLOR = "#aaaaaa"


def _share_cache_key(share_id: str) -> str:
    return f"share:{share_id}"


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


def _render_chart(share: dict, records: list) -> bytes:
    """Render a temperature-over-years line chart as PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unit = share.get("unit", "celsius")
    ref_year = share.get("ref_year")
    location = share.get("location", "")
    period = share.get("period", "").capitalize()
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

    fig, ax = plt.subplots(figsize=(_IMG_W / 100, _IMG_H / 100), dpi=100)
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_AXES)

    # Historical average (excludes ref_year)
    hist_temps = [t for y, t in zip(years, temps) if y != ref_year]
    if hist_temps:
        avg = sum(hist_temps) / len(hist_temps)
        ax.axvline(
            avg,
            color=_AVG_LINE,
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            zorder=1,
            label=f"Avg {avg:.1f}{unit_symbol}",
        )

    # Horizontal bars: ref_year in green, historical years in red
    bar_colors = [_REF_YEAR if y == ref_year else _BAR for y in years]
    ax.barh(years, temps, color=bar_colors, height=0.7, zorder=2, alpha=0.85)

    # Annotate ref_year bar with its value
    if ref_year in years:
        idx = years.index(ref_year)
        ref_temp = temps[idx]
        ax.annotate(
            f"{ref_temp:.1f}{unit_symbol}",
            xy=(ref_temp, ref_year),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=_REF_YEAR,
            fontsize=13,
            fontweight="bold",
        )

    # Labels and styling
    ax.set_title(
        f"{location}  ·  {period} {identifier}",
        color="white",
        fontsize=17,
        pad=14,
        fontweight="bold",
    )
    ax.set_xlabel(f"Temperature ({unit_symbol})", color=_TICK_COLOR, fontsize=12)
    ax.set_ylabel("Year", color=_TICK_COLOR, fontsize=12)
    ax.tick_params(colors=_TICK_COLOR, which="both")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    if hist_temps:
        ax.legend(facecolor=_BG_AXES, edgecolor="#333355", labelcolor=_TICK_COLOR, fontsize=11)

    plt.tight_layout(pad=1.8)

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

    location = share.get("location", "")
    period = share.get("period", "").capitalize()
    raw_identifier = share.get("identifier", "")
    # Identifier is stored as MM-DD; display as DD-MM for UK/European format
    parts = raw_identifier.split("-")
    identifier = f"{parts[1]}-{parts[0]}" if len(parts) == 2 else raw_identifier
    ref_year = share.get("ref_year", "")

    fig, ax = plt.subplots(figsize=(_IMG_W / 100, _IMG_H / 100), dpi=100)
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_DARK)
    ax.axis("off")

    ax.text(
        0.5, 0.62, "TempHist",
        ha="center", va="center", transform=ax.transAxes,
        color="white", fontsize=54, fontweight="bold",
    )
    ax.text(
        0.5, 0.44, location,
        ha="center", va="center", transform=ax.transAxes,
        color=_AVG_LINE, fontsize=24,
    )
    ax.text(
        0.5, 0.32, f"{period}  ·  {identifier}  ·  {ref_year}",
        ha="center", va="center", transform=ax.transAxes,
        color=_TICK_COLOR, fontsize=16,
    )

    plt.tight_layout(pad=1.5)
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
