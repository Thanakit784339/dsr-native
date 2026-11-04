#!/usr/bin/env python3
"""Minimal Switch NSO0 loader: decompress segments, build flat image, parse MOD0/.dynamic/.dynsym."""
import struct, sys, hashlib, json, os

def lz4_block(src, out_size):
    dst = bytearray(); i = 0; n = len(src)
    while i < n:
        tok = src[i]; i += 1
        lit = tok >> 4
        if lit == 15:
            while True:
                b = src[i]; i += 1; lit += b
                if b != 255: break
        dst += src[i:i+lit]; i += lit
        if i >= n: break
        off = src[i] | (src[i+1] << 8); i += 2
        ml = tok & 15
        if ml == 15:
            while True:
                b = src[i]; i += 1; ml += b
                if b != 255: break
        ml += 4
        start = len(dst) - off
        if off >= ml:
            dst += dst[start:start+ml]
        else:
            for k in range(ml): dst.append(dst[start+k])
    assert len(dst) == out_size, (len(dst), out_size)
    return bytes(dst)

def load(path):
    d = open(path,'rb').read()
    assert d[:4] == b'NSO0'
    flags = struct.unpack_from('<I', d, 0xC)[0]
    segs = []
    for k, name in enumerate(('text','rodata','data')):
        foff, moff, size = struct.unpack_from('<III', d, 0x10 + k*0x10)
        csize = struct.unpack_from('<I', d, 0x60 + k*4)[0]
        raw = d[foff:foff+(csize if flags & (1<<k) else size)]
        data = lz4_block(raw, size) if flags & (1<<k) else raw
        if flags & (1<<(k+3)):
            h = d[0xA0+k*0x20:0xC0+k*0x20]
            assert hashlib.sha256(data).digest() == h, f"hash mismatch {name}"
        segs.append(dict(name=name, moff=moff, size=size, data=data))
    bss = struct.unpack_from('<I', d, 0x3C)[0]
    build_id = d[0x40:0x60].hex()
    end = segs[2]['moff'] + segs[2]['size'] + bss
    img = bytearray(end)
    for s in segs: img[s['moff']:s['moff']+s['size']] = s['data']
    return dict(segs=segs, bss=bss, build_id=build_id, image=bytes(img), flags=flags)

def parse_dyn(img):
    mod0 = struct.unpack_from('<I', img, 4)[0]
    assert img[mod0:mod0+4] == b'MOD0', 'no MOD0'
    dyn_off, bss_s, bss_e, eh_s, eh_e, modobj = struct.unpack_from('<iiiiii', img, mod0+4)
    dyn = mod0 + dyn_off
    tags = {}; i = dyn
    while True:
        t, v = struct.unpack_from('<qQ', img, i); i += 16
        if t == 0: break
        tags.setdefault(t, []).append(v)
    strtab = tags[5][0]; symtab = tags[6][0]
    # nsyms via hash table (DT_HASH=4) or GNU hash
    nsyms = None
    if 4 in tags: nsyms = struct.unpack_from('<I', img, tags[4][0]+4)[0]
    else: nsyms = (strtab - symtab)//24
    def cstr(o):
        e = img.index(b'\0', o); return img[o:e].decode('utf-8','replace')
    syms = []
    for k in range(nsyms):
        nm, info, oth, shn, val, sz = struct.unpack_from('<IBBHQQ', img, symtab + k*24)
        syms.append(dict(name=cstr(strtab+nm), type=info & 0xf, bind=info>>4, shndx=shn, value=val, size=sz))
    needed = [cstr(strtab+v) for v in tags.get(1, [])]
    rela = []
    for (a,s,cnt_tag) in ((7,8,9),(23,2,None)):
        if a in tags:
            base = tags[a][0]; size = tags[s][0]
            for o in range(base, base+size, 24):
                off, info, add = struct.unpack_from('<QQq', img, o)
                rela.append((off, info & 0xffffffff, info >> 32, add))
    return dict(mod0=mod0, dynamic=dyn, eh_frame_hdr=(mod0+eh_s, mod0+eh_e), bss=(mod0+bss_s, mod0+bss_e),
                tags={k:v for k,v in tags.items()}, syms=syms, needed=needed, rela=rela)

if __name__ == '__main__':
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    n = load(src); base = os.path.basename(src)
    open(os.path.join(outdir, base+'.bin'),'wb').write(n['image'])
    info = parse_dyn(n['image'])
    defined = [s for s in info['syms'] if s['shndx'] and s['value']]
    imported = [s for s in info['syms'] if not s['shndx'] and s['name']]
    summary = dict(file=base, build_id=n['build_id'],
        segments=[dict(name=s['name'], off=hex(s['moff']), size=hex(s['size'])) for s in n['segs']],
        bss=hex(n['bss']), image_size=hex(len(n['image'])), needed=info['needed'],
        n_syms=len(info['syms']), n_defined=len(defined), n_defined_funcs=sum(1 for s in defined if s['type']==2),
        n_imported=len(imported), n_relocs=len(info['rela']), eh_frame_hdr=[hex(x) for x in info['eh_frame_hdr']])
    print(json.dumps(summary, indent=1))
    with open(os.path.join(outdir, base+'.syms.tsv'),'w') as f:
        for s in info['syms']:
            f.write(f"{s['value']:#x}\t{s['size']}\t{'FUNC' if s['type']==2 else 'OBJ' if s['type']==1 else s['type']}\t{'DEF' if s['shndx'] else 'IMP'}\t{s['name']}\n")
