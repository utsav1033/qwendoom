"""Cut a Doom IWAD down to what one demo needs, for the Qwenputer machine's DROM.

Every lump stays in place (Doom resolves a name to the last lump of that name). Keeps the demo
and its map (and every other map's 0-byte marker: Doom tells the game mode from which exist); the
wall textures the map uses, plus every texture the engine looks
up by name at startup (switches, animations, the episode's sky), in their original order (an
animation is the textures between its first and last); PNAMES and TEXTURE1 rebuilt to just those,
and the patches they use; the map's flats plus F_SKY1 and every flat animation the map touches (every other flat stays as
an empty lump, so the flats keep the full file's count and numbers);
all sprites; and every other lump but sounds, other maps and other demos (music lumps stay,
emptied: they are looked up by name). The engine is
unchanged: it reads this file exactly as it reads the full IWAD.

Run: python port/wad.py freedoom1.wad port/build/doom.wad [DEMO1]   (doomgeneric fetched by
port/build.sh, for the switch and animation tables)
"""

import re
import struct
import sys
from pathlib import Path

DG = Path(__file__).resolve().parent / "build" / "doomgeneric" / "doomgeneric"
MAP_LUMPS = ("THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS", "SSECTORS", "NODES",
             "SECTORS", "REJECT", "BLOCKMAP")


def read(path):
    b = Path(path).read_bytes()
    n, off = struct.unpack_from("<4xII", b)
    out = []
    for i in range(n):
        p, s, nm = struct.unpack_from("<II8s", b, off + 16 * i)
        out.append((nm.rstrip(b"\0").decode("ascii").upper(), b[p:p + s]))
    return out


def write(path, lumps):
    data, dirs = bytearray(), bytearray()
    for nm, blob in lumps:
        dirs += struct.pack("<II8s", 12 + len(data), len(blob), nm.encode())
        data += blob + bytes(-len(blob) % 4)
    Path(path).write_bytes(struct.pack("<4sII", b"IWAD", len(lumps), 12 + len(data)) + data + dirs)


def name8(b, o=0):
    return b[o:o + 8].split(b"\0")[0].decode("ascii").upper()


def engine_tables():
    """(switch texture names, [(is_texture, start, end)] animations) from doomgeneric's tables."""
    sw = (DG / "p_switch.c").read_text()
    sw = sw[sw.index("alphSwitchList[]"):]
    sw = sw[:sw.index("};")]
    switches = set()
    for a, b, ep in re.findall(r'\{\s*"(\w+)",\s*"(\w+)",\s*(\d+)\s*\}', sw):
        if int(ep) <= 2:  # Ultimate Doom mode (Freedoom: Phase 1) loads episodes 1 and 2's
            switches |= {a.upper(), b.upper()}
    an = (DG / "p_spec.c").read_text()
    an = an[an.index("animdefs[]"):]
    an = an[:an.index("};")]
    anims = [(t == "true", s.upper(), e.upper())
             for t, e, s in re.findall(r'\{\s*(true|false),\s*"(\w+)",\s*"(\w+)"', an)]
    return switches, anims


def main(src, dst, demo="DEMO1"):
    lumps = read(src)
    names = [n for n, _ in lumps]
    get = {n: b for n, b in lumps}
    d = get[demo]
    assert d[0] == 109, "a Doom 1.9 demo"
    mapname = f"E{d[2]}M{d[3]}"
    i = names.index(mapname)
    maplumps = lumps[i:i + 1 + len(MAP_LUMPS)]
    assert [n for n, _ in maplumps[1:]] == list(MAP_LUMPS)
    m = dict(maplumps[1:])

    switches, anims = engine_tables()
    walls = set()
    sd = m["SIDEDEFS"]
    for o in range(0, len(sd), 30):
        walls |= {name8(sd, o + 4), name8(sd, o + 12), name8(sd, o + 20)}
    walls.discard("-")
    floors = {"F_SKY1"}
    se = m["SECTORS"]
    for o in range(0, len(se), 26):
        floors |= {name8(se, o + 4), name8(se, o + 12)}

    # textures: TEXTURE1 (+ TEXTURE2) in order; keep the used ones, switches, the sky, and whole
    # animations any of whose textures is kept
    pnames_b = get["PNAMES"]
    pnames = [name8(pnames_b, 4 + 8 * k) for k in range(struct.unpack_from("<i", pnames_b)[0])]
    texs = []
    for tl in ("TEXTURE1", "TEXTURE2"):
        if tl in get:
            t = get[tl]
            cnt = struct.unpack_from("<i", t)[0]
            for k in range(cnt):
                o = struct.unpack_from("<i", t, 4 + 4 * k)[0]
                npatch = struct.unpack_from("<h", t, o + 20)[0]
                texs.append((name8(t, o), t[o:o + 22 + 10 * npatch]))
    order = [n for n, _ in texs]
    keep_t = walls | switches | {"SKY1"}
    for is_tex, s, e in anims:
        if is_tex and s in order and e in order:
            lo, hi = order.index(s), order.index(e)
            if any(n in keep_t for n in order[lo:hi + 1]):
                keep_t |= set(order[lo:hi + 1])
    kept = [(n, b) for n, b in texs if n in keep_t]
    missing = keep_t - {n for n, _ in kept}
    assert not missing - switches, f"textures not in the IWAD: {missing - switches}"
    used = []  # patches, in first-use order
    for _, b in kept:
        for k in range(struct.unpack_from("<h", b, 20)[0]):
            p = pnames[struct.unpack_from("<h", b, 22 + 10 * k + 4)[0]]
            if p not in used:
                used.append(p)
    pidx = {p: k for k, p in enumerate(used)}
    tex1 = bytearray(struct.pack("<i", len(kept)))
    body, offs = bytearray(), []
    for _, b in kept:
        b = bytearray(b)
        for k in range(struct.unpack_from("<h", b, 20)[0]):
            o = 22 + 10 * k + 4
            struct.pack_into("<h", b, o, pidx[pnames[struct.unpack_from("<h", b, o)[0]]])
        offs.append(len(body))
        body += b
    base = 4 + 4 * len(kept)
    tex1 += b"".join(struct.pack("<i", base + o) for o in offs) + body
    pn = struct.pack("<i", len(used)) + b"".join(p.encode().ljust(8, b"\0") for p in used)

    # flats: the map's, F_SKY1, and whole flat animations the map touches, in lump order
    fs, fe = names.index("F_START"), names.index("F_END")
    flat_order = names[fs + 1:fe]  # inner markers too: they count in a flat animation's range
    for is_tex, s, e in anims:
        if not is_tex and s in flat_order and e in flat_order:
            lo, hi = flat_order.index(s), flat_order.index(e)
            if any(n in floors for n in flat_order[lo:hi + 1]):
                floors |= set(flat_order[lo:hi + 1])

    # every lump stays where it was (Doom resolves a name to its last lump, so order matters);
    # dropped ones go, a few are emptied, PNAMES and TEXTURE1 are replaced in place
    out, section, inmap = [], None, None
    for nm, b in lumps:
        if inmap and nm in MAP_LUMPS:
            if inmap == mapname:
                out.append((nm, b))
            continue
        inmap = None
        if nm.endswith(("_START", "_END")):
            if nm in ("S_START", "F_START", "P_START"):
                section = nm[0]
            elif nm in ("S_END", "F_END", "P_END"):
                section = None
            out.append((nm, b))  # 0 bytes; a flat animation counts inner markers as frames
        elif section == "S":
            out.append((nm, b))
        elif section == "F":  # every flat keeps its place (the flat count and numbering are
            out.append((nm, b if nm in floors else b""))  # the full file's); unused ones are empty
        elif section == "P":
            if nm in pidx:
                out.append((nm, b))
        elif nm == "PNAMES":
            out.append((nm, pn))
        elif nm == "TEXTURE1":
            out.append((nm, bytes(tex1)))
        elif nm == "TEXTURE2":
            continue  # merged into TEXTURE1, in order
        elif re.fullmatch(r"E\dM\d", nm):
            inmap = nm
            out.append((nm, b))  # a marker, 0 bytes: the game mode comes from which exist
        elif re.fullmatch(r"DEMO\d", nm):
            if nm == demo:
                out.append((nm, b))
        elif nm.startswith(("DS", "DP")):
            continue  # sound effects, PC speaker sounds: no sound on the machine
        elif nm.startswith("D_"):
            out.append((nm, b""))  # music: looked up by name at every level start, never played
        else:
            out.append((nm, b))
    write(dst, out)
    size = Path(dst).stat().st_size
    print(f"{dst}: {mapname} for {demo}, {len(kept)} textures, {len(used)} patches, "
          f"{sum(1 for n in floors if n in flat_order)} flats, {len(out)} lumps, "
          f"{size / 1e6:.2f} MB (from {Path(src).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
