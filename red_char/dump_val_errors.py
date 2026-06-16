"""Render the deployment reranker's validation errors as images for inspection.

Saves: (1) a montage of full error captchas with true/pred titles, (2) a montage
of just the misclassified glyph crops. Lets us judge whether errors are
human-readable (model is fixable) or genuinely ambiguous.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as VF

import config
from dataset import build_train_dataset, decode_prediction, deterministic_split_indices
from glyph import glyph_probabilities, load_glyph_model, extract_glyph_crops, GLYPH_CROP_WIDTH
from predict import load_model
from eval_reranker import selective_rerank


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--glyph-checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--primary-margin-max", type=float, default=0.20)
    p.add_argument("--glyph-margin-min", type=float, default=0.05)
    p.add_argument("--red-threshold", type=float, default=0.20)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--outdir", type=Path, default=config.EDA_DIR / "deploy_errors")
    args = p.parse_args()

    device = torch.device(config.DEVICE)
    base = build_train_dataset(cache_in_ram=False)
    _, val_idx = deterministic_split_indices(len(base))
    val_names = [base.samples[i].filename for i in val_idx]
    loader = DataLoader(Subset(base, val_idx), batch_size=256, shuffle=False, num_workers=4)
    prim = [load_model(c, device) for c in args.checkpoints]
    glyph = [load_glyph_model(c, device) for c in args.glyph_checkpoints]

    P=[];C=[];G=[];CT=[];KT=[];IMG=[]
    for images, ct, kt in loader:
        images = images.to(device)
        pc=kp=None
        for shift in (0,-4,4):
            sh = images if shift==0 else VF.affine(images,angle=0,translate=[shift,0],scale=1.0,shear=[0.,0.],interpolation=InterpolationMode.BILINEAR,fill=[1.,1.,1.])
            for m in prim:
                cl,kl=m(sh); a,b=F.softmax(cl,-1),F.softmax(kl,-1)
                pc=a if pc is None else pc+a; kp=b if kp is None else kp+b
        div=len(prim)*3
        P.append((pc/div).cpu()); C.append((kp/div).cpu())
        G.append(glyph_probabilities(glyph,images).cpu())
        CT.append(ct); KT.append(kt); IMG.append(images.cpu())
    P=torch.cat(P);C=torch.cat(C);G=torch.cat(G);CT=torch.cat(CT);KT=torch.cat(KT);IMG=torch.cat(IMG)

    char_pred = selective_rerank(P, G, args.top_k, args.primary_margin_max, args.glyph_margin_min)
    color_pred = C[...,config.RED_INDEX].ge(args.red_threshold).long()

    errs=[]
    for i in range(P.shape[0]):
        ts=decode_prediction(CT[i].tolist(),KT[i].tolist())
        ps=decode_prediction(char_pred[i].tolist(),color_pred[i].tolist())
        if ps!=ts:
            wrong=[pos for pos in range(5) if (KT[i,pos]==config.RED_INDEX and char_pred[i,pos]!=CT[i,pos]) or KT[i,pos]!=color_pred[i,pos]]
            errs.append((i,ts,ps,wrong))
    print(f"deployment reranker val errors: {len(errs)}/{P.shape[0]}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    # full-image montage with titles
    cols=6; rows=(len(errs)+cols-1)//cols
    cell_w, cell_h = 200, 60+16
    sheet=Image.new("RGB",(cols*cell_w, rows*cell_h),(255,255,255))
    draw=ImageDraw.Draw(sheet)
    crop_imgs=[]
    for j,(i,ts,ps,wrong) in enumerate(errs):
        arr=(IMG[i].permute(1,2,0).numpy()*255).astype(np.uint8)
        im=Image.fromarray(arr)
        r,c=divmod(j,cols)
        sheet.paste(im,(c*cell_w, r*cell_h+16))
        draw.text((c*cell_w+2, r*cell_h+2), f"T:{ts} P:{ps}", fill=(200,0,0))
        # save wrong-position crops at high zoom
        crops=extract_glyph_crops(IMG[i].unsqueeze(0))[0]  # [5,3,60,64]
        for pos in wrong:
            ca=(crops[pos].permute(1,2,0).numpy()*255).astype(np.uint8)
            cim=Image.fromarray(ca).resize((128,120),Image.NEAREST)
            tc=config.IDX_TO_CHAR[int(CT[i,pos])]; pc_=config.IDX_TO_CHAR[int(char_pred[i,pos])]
            tag=f"T{tc}P{pc_}" if KT[i,pos]==config.RED_INDEX else f"col{('u','r')[int(KT[i,pos])]}->{('u','r')[int(color_pred[i,pos])]}"
            crop_imgs.append((cim,tag))
    sheet.save(args.outdir/"errors_full.png")
    # crop montage
    if crop_imgs:
        cc=8; cr=(len(crop_imgs)+cc-1)//cc
        cs=Image.new("RGB",(cc*128, cr*140),(255,255,255))
        cd=ImageDraw.Draw(cs)
        for j,(cim,tag) in enumerate(crop_imgs):
            r,c=divmod(j,cc); cs.paste(cim,(c*128,r*140+18)); cd.text((c*128+2,r*140+2),tag,fill=(200,0,0))
        cs.save(args.outdir/"errors_crops.png")
    print("saved:", args.outdir/"errors_full.png", "and errors_crops.png")
    for i,ts,ps,wrong in errs:
        print(f"  {val_names[i]} T={ts} P={ps} wrong_pos={wrong}")


if __name__ == "__main__":
    main()
