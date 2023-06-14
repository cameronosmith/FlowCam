from run import *

# Evaluation script
import piqa,lpips
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
lpips,psnr,ate=0,0,0

eval_dir = save_dir+"/"+args.name+datetime.datetime.now().strftime("%b%d%Y_")+str(random.randint(0,1e3))
try: os.mkdir(eval_dir)
except: pass
torch.set_grad_enabled(False)

model.n_samples=128

val_dataset = get_dataset(val=True,)

for eval_idx,eval_dataset_idx in enumerate(tqdm(torch.linspace(0,len(val_dataset)-1,min(args.n_eval,len(val_dataset))).int())):
    model_input,ground_truth = val_dataset[eval_dataset_idx]

    for x in (model_input,ground_truth): 
        for k,v in x.items(): x[k] = v[None].cuda() # collate

    model_out = model.render_full_img(model_input)

    # remove last frame since used as ctxt when n_ctxt=2
    rgb_est,rgb_gt = [rearrange(img[:,:-1].clip(0,1),"b trgt (x y) c -> (b trgt) c x y",x=model_input["trgt_rgb"].size(-2)) 
                                            for img in (model_out["fine_rgb" if "fine_rgb" in model_out else "rgb"],ground_truth["trgt_rgb"])]
    depth_est = rearrange(model_out["depth"][:,:-1],"b trgt (x y) c -> (b trgt) c x y",x=model_input["trgt_rgb"].size(-2))

    psnr += piqa.PSNR()(rgb_est.clip(0,1).contiguous(),rgb_gt.clip(0,1).contiguous())
    lpips += loss_fn_vgg(rgb_est*2-1,rgb_gt*2-1).mean()

    print(args.save_imgs)
    if args.save_imgs:
        fp = os.path.join(eval_dir,f"{eval_idx}_est.png");plt.imsave(fp,make_grid(rgb_est).permute(1,2,0).clip(0,1).cpu().numpy())
        if depth_est.size(1)==3: fp = os.path.join(eval_dir,f"{eval_idx}_depth.png");plt.imsave(fp,make_grid(depth_est).clip(0,1).permute(1,2,0).cpu().numpy())
        fp = os.path.join(eval_dir,f"{eval_idx}_gt.png");plt.imsave(fp,make_grid(rgb_gt).permute(1,2,0).cpu().numpy())
        print(fp)


    if args.save_imgs and args.save_ind: # save individual images separately
        eval_idx_dir = os.path.join(eval_dir,f"dir_{eval_idx}")

        try: os.mkdir(eval_idx_dir)
        except: pass
        ctxt_rgbs = torch.cat((model_input["ctxt_rgb"][:,0],model_input["trgt_rgb"][:,model_input["trgt_rgb"].size(1)//2],model_input["trgt_rgb"][:,-1]))*.5+.5
        fp = os.path.join(eval_idx_dir,f"ctxt0.png");plt.imsave(fp,ctxt_rgbs[0].clip(0,1).permute(1,2,0).cpu().numpy())
        fp = os.path.join(eval_idx_dir,f"ctxt1.png");plt.imsave(fp,ctxt_rgbs[1].clip(0,1).permute(1,2,0).cpu().numpy())
        fp = os.path.join(eval_idx_dir,f"ctxt2.png");plt.imsave(fp,ctxt_rgbs[2].clip(0,1).permute(1,2,0).cpu().numpy())
        for i,(rgb_est,rgb_gt,depth) in enumerate(zip(rgb_est,rgb_gt,depth_est)):
            fp = os.path.join(eval_idx_dir,f"{i}_est.png");plt.imsave(fp,rgb_est.clip(0,1).permute(1,2,0).cpu().numpy())
            print(fp)
            fp = os.path.join(eval_idx_dir,f"{i}_gt.png");plt.imsave(fp,rgb_gt.clip(0,1).permute(1,2,0).cpu().numpy())
            if depth_est.size(1)==3: fp = os.path.join(eval_idx_dir,f"{i}_depth.png");plt.imsave(fp,depth.permute(1,2,0).cpu().clip(1e-4,1-1e-4).numpy())

    # Pose plotting/evaluation
    if "poses" in model_out:
        import scipy.spatial
        pose_est,pose_gt = model_out["poses"][0][:,:3,-1].cpu(),model_input["trgt_c2w"][0][:,:3,-1].cpu()
        pose_gt,pose_est,_ = scipy.spatial.procrustes(pose_gt.numpy(),pose_est.numpy())
        ate += ((pose_est-pose_gt)**2).mean()
        if args.save_imgs:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(*pose_gt.T)
            ax.plot(*pose_est.T)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.zaxis.set_tick_params(labelleft=False)
            ax.view_init(elev=10., azim=45)
            plt.tight_layout()
            fp = os.path.join(eval_dir,f"{eval_idx}_pose_plot.png");plt.savefig(fp,bbox_inches='tight');plt.close()
            if args.save_ind:
                for i in range(len(pose_est)):
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot(*pose_gt.T,color="black")
                    ax.plot(*pose_est.T,alpha=0)
                    ax.plot(*pose_est[:i].T,alpha=1,color="red")
                    ax.xaxis.set_tick_params(labelbottom=False)
                    ax.yaxis.set_tick_params(labelleft=False)
                    ax.zaxis.set_tick_params(labelleft=False)
                    ax.view_init(elev=10., azim=45)
                    plt.tight_layout()
                    fp = os.path.join(eval_idx_dir,f"pose_{i}.png"); plt.savefig(fp,bbox_inches='tight');plt.close()

    print(f"psnr {psnr/(1+eval_idx)}, lpips {lpips/(1+eval_idx)}, ate {(ate/(1+eval_idx))**.5}, eval_idx {eval_idx}", flush=True)

