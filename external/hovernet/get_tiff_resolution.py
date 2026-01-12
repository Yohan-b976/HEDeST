import re
from fractions import Fraction
import tifffile

def rational_to_float(r):
    # tifffile gives rational as (num, den) or Fraction
    if isinstance(r, Fraction):
        return float(r)
    try:
        num, den = r
        return float(num) / float(den)
    except Exception:
        try:
            return float(r)
        except Exception:
            return None

def parse_unit_from_image_description(desc):
    if not desc:
        return None
    # common forms: unit=µm, unit=\u00B5m, unit=um
    m = re.search(r"unit\s*=\s*(?:\\u00B5m|µm|um)", desc, flags=re.IGNORECASE)
    if m:
        return "um"
    # sometimes ImageDescription contains "unit=micron" or "unit=micrometer"
    m2 = re.search(r"unit\s*=\s*(micron|micrometer)", desc, flags=re.IGNORECASE)
    if m2:
        return "um"
    return None

def search_for_pixel_size_in_text(text):
    if not text:
        return None
    # find floats between 0.1 and 10 (reasonable µm/pixel range typical for microscopy)
    floats = re.findall(r"([0-9]+(?:\.[0-9]+)?)", text)
    candidates = []
    for f in floats:
        try:
            v = float(f)
        except:
            continue
        # consider plausible pixel sizes in microns
        if 0.01 <= v <= 100:     # generous bounds
            candidates.append(v)
    # return smallest plausible (often pixel size is small)
    return min(candidates) if candidates else None

def extract_mpp(path):
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = page.tags

        # 1) Try tif.imagej_metadata (best)
        ij_meta = getattr(tif, "imagej_metadata", None) or getattr(page, "imagej_metadata", None)
        if ij_meta:
            # imagej_metadata is usually a dict
            pxw = ij_meta.get("pixelWidth") or ij_meta.get("spacing") or ij_meta.get("XResolution")
            pxh = ij_meta.get("pixelHeight") or ij_meta.get("spacing") or ij_meta.get("YResolution")
            unit = ij_meta.get("unit")
            if pxw:
                try:
                    mpp_x = float(pxw)
                    mpp_y = float(pxh) if pxh else mpp_x
                    return {"mpp_x": mpp_x, "mpp_y": mpp_y, "unit": unit or "unknown", "source": "imagej_metadata"}
                except Exception:
                    pass

        # 2) ImageDescription + X/YResolution with ResolutionUnit == NONE
        img_desc = tags.get("ImageDescription")
        desc_val = img_desc.value if img_desc else ""
        unit = parse_unit_from_image_description(desc_val)

        xres_tag = tags.get("XResolution")
        yres_tag = tags.get("YResolution")
        res_unit_tag = tags.get("ResolutionUnit")
        xres = rational_to_float(xres_tag.value) if xres_tag else None
        yres = rational_to_float(yres_tag.value) if yres_tag else None
        res_unit = res_unit_tag.value if res_unit_tag else None

        # If ResolutionUnit == NONE and unit==um, interpret XResolution as pixels-per-µm
        if res_unit == 1 or res_unit == "NONE":   # some libs give numeric code 1 for NONE
            if unit == "um" and xres:
                try:
                    mpp_x = 1.0 / xres
                    mpp_y = (1.0 / yres) if yres else mpp_x
                    return {"mpp_x": mpp_x, "mpp_y": mpp_y, "unit": "µm", "source": "x/y resolution (pixels per µm)"}
                except Exception:
                    pass

        # 3) Try to parse IJMetadata tag bytes (TAG 50839 / IJMetadata)
        ijtag = tags.get(50839) or tags.get("IJMetadata")
        if ijtag:
            raw = ijtag.value
            # raw sometimes a bytes array or a dict-like repr; convert to string
            s = None
            try:
                if isinstance(raw, (bytes, bytearray)):
                    s = raw.decode("utf-8", errors="ignore")
                else:
                    s = str(raw)
            except:
                s = str(raw)
            found = search_for_pixel_size_in_text(s)
            if found:
                return {"mpp_x": found, "mpp_y": found, "unit": "µm (guessed from IJMetadata text)", "source": "IJMetadata text heuristic"}

        # 4) As a last resort, if ResolutionUnit is INCH or CENTIMETER interpret as DPI
        if res_unit in (2, "INCH", "INCHES") and xres:
            # dpi -> mpp = 25400 microns / dpi
            mpp_x = 25400.0 / xres
            mpp_y = 25400.0 / yres if yres else mpp_x
            return {"mpp_x": mpp_x, "mpp_y": mpp_y, "unit": "µm", "source": "dpi conversion (INCH)"}
        if res_unit in (3, "CENTIMETER", "CENTIMETERS") and xres:
            # dpi per cm -> 10000 µm / (res per cm)
            mpp_x = 10000.0 / xres
            mpp_y = 10000.0 / yres if yres else mpp_x
            return {"mpp_x": mpp_x, "mpp_y": mpp_y, "unit": "µm", "source": "dpi conversion (CM)"}

        # 5) If XResolution is a small number (< 20) and no explicit unit, guess it's pixels-per-µm
        if xres and xres < 50:
            # heuristic: many microscopy images have pixels-per-micron in single digits
            try:
                mpp_x = 1.0 / xres
                mpp_y = (1.0 / yres) if yres and yres < 50 else mpp_x
                return {"mpp_x": mpp_x, "mpp_y": mpp_y, "unit": "µm (heuristic: reciprocal of XResolution)", "source": "heuristic reciprocal"}
            except Exception:
                pass

        return {"mpp_x": None, "mpp_y": None, "unit": None, "source": "not found"}