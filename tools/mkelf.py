#!/usr/bin/env python3
"""Wrap a decompressed NSO image into an AArch64 ELF (ET_DYN, base 0) with .text/.rodata/.data/.bss
section headers and a .symtab containing exported symbols plus sub_<addr> for every eh_frame FDE start."""
import struct, sys, json
sys.path.insert(0, __file__.rsplit('/',1)[0])
import nso
src, funcs_txt, out = sys.argv[1:4]
n = nso.load(src); img = n['image']; info = nso.parse_dyn(img)
segs = n['segs']; bss_start = segs[2]['moff']+segs[2]['size']; bss_size = n['bss']
named = {}
for s in info['syms']:
    if s['shndx'] and s['value']: named.setdefault(s['value'], s)
starts = [int(l,16) for l in open(funcs_txt)]
syms = []  # (name, value, size, type, shidx)
text_end = segs[0]['size']
for i, a in enumerate(starts):
    nxt = starts[i+1] if i+1 < len(starts) else text_end
    if a in named and named[a]['type']==2: syms.append((named[a]['name'], a, named[a]['size'] or nxt-a, 2, 1))
    else: syms.append((f"sub_{a:x}", a, max(0, min(nxt, text_end)-a), 2, 1))
# Name PLT stubs after the import they jump to: adrp x16, page; ldr x17, [x16, #off]; add x16..; br x17
got_to_name = {r[0]: info['syms'][r[2]]['name'] for r in info['rela'] if r[1] == 0x402}
text = img[:segs[0]['size']]
plt_named = 0
for o in range(0, len(text) - 16, 4):
    w0, w1, w2, w3 = struct.unpack_from('<IIII', text, o)
    if (w0 & 0x9f00001f) == 0x90000010 and (w1 & 0xffc003ff) == 0xf9400211 and w3 == 0xd61f0220:
        immlo = (w0 >> 29) & 3; immhi = (w0 >> 5) & 0x7ffff
        imm = ((immhi << 2) | immlo)
        if imm & (1 << 20): imm -= 1 << 21
        page = (o & ~0xfff) + (imm << 12)
        got = page + ((w1 >> 10) & 0xfff) * 8
        nm = got_to_name.get(got)
        if nm:
            syms.append((nm, o, 16, 2, 1)); plt_named += 1  # bare mangled name so disassemblers demangle it
print('named PLT stubs:', plt_named)
seen = {a for _,a,_,_,_ in syms}
for v, s in named.items():
    if v not in seen:
        sec = 1 if v < segs[1]['moff'] else 2 if v < segs[2]['moff'] else 3
        syms.append((s['name'], v, s['size'], s['type'] if s['type'] in (1,2) else 0, sec))
shstr = b'\0.text\0.rodata\0.data\0.bss\0.symtab\0.strtab\0.shstrtab\0'
def soff(n): return shstr.index(n.encode()+b'\0')
strtab = bytearray(b'\0'); symtab = bytearray(b'\0'*24)
for name, v, sz, t, sh in sorted(syms, key=lambda x: x[1]):
    off = len(strtab); strtab += name.encode()+b'\0'
    symtab += struct.pack('<IBBHQQ', off, (1<<4)|t, 0, sh, v, sz)
body = bytearray(0x40); fileoff_img = 0x1000
body += b'\0'*(fileoff_img-len(body)); body += img[:bss_start]
def align(b,a=8): b += b'\0'*((-len(b))%a)
align(body); sym_off = len(body); body += symtab
str_off = len(body); body += strtab
shs_off = len(body); body += shstr; align(body)
# program headers: put after header? keep simple: phdrs at end
ph_off = len(body)
flagsmap = [5,4,6]
for i,s in enumerate(segs):
    memsz = s['size'] + (bss_size if i==2 else 0)
    body += struct.pack('<IIQQQQQQ', 1, flagsmap[i], fileoff_img+s['moff'], s['moff'], s['moff'], s['size'], memsz, 0x1000)
sh_off = len(body)
def sh(name, typ, flags, addr, off, size, link=0, info=0, align=1, entsize=0):
    return struct.pack('<IIQQQQIIQQ', name, typ, flags, addr, off, size, link, info, align, entsize)
shdrs = [sh(0,0,0,0,0,0)]
shdrs.append(sh(soff('.text'),1,6,0,fileoff_img,segs[0]['size'],align=4))
shdrs.append(sh(soff('.rodata'),1,2,segs[1]['moff'],fileoff_img+segs[1]['moff'],segs[1]['size'],align=8))
shdrs.append(sh(soff('.data'),1,3,segs[2]['moff'],fileoff_img+segs[2]['moff'],segs[2]['size'],align=8))
shdrs.append(sh(soff('.bss'),8,3,bss_start,0,bss_size,align=8))
shdrs.append(sh(soff('.symtab'),2,0,0,sym_off,len(symtab),link=6,info=1,align=8,entsize=24))
shdrs.append(sh(soff('.strtab'),3,0,0,str_off,len(strtab)))
shdrs.append(sh(soff('.shstrtab'),3,0,0,shs_off,len(shstr)))
for h in shdrs: body += h
hdr = b'\x7fELF' + bytes([2,1,1,0]) + b'\0'*8
hdr += struct.pack('<HHIQQQIHHHHHH', 3, 183, 1, 0, ph_off, sh_off, 0, 64, 56, 3, 64, len(shdrs), 7)
body[0:0x40] = hdr
open(out,'wb').write(body)
print(f"{out}: {len(syms)} symbols")
