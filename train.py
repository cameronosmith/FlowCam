# todo get rid of start import, just import run and access run.train_dataset, etc 
from run import *

train_dataset,val_dataset = get_dataset(),get_dataset(val=True,)
train_dataset[0]
train_dataloader,val_dataloader = get_dataloader(train_dataset),get_dataloader(val_dataset)

def loss_fn(model_out, gt, model_input):

    rays = lambda x,y: torch.stack([x[i,:,y[i].long()] for i in range(len(x))])
    ch_sec = lambda x: rearrange(x,"... c x y -> ... (x y) c")
    ch_fst = lambda src,x=None:rearrange(src,"... (x y) c -> ... c x y",x=int(src.size(-2)**(.5)) if x is None else x)
    losses = { "metrics/rgb_loss": (model_out["rgb"] - rays(gt["trgt_rgb"],model_out["trgt_rays"])).square().mean() * (1e1 if args.dataset=="shapenet" else 2e2) }

    losses["metrics/ctxt_rgb_loss"]= (model_out["ctxt_rgb"] - rays(gt["ctxt_rgb"],model_out["ctxt_rays"])).square().mean() * 1e2
    gt_bwd_flow = rays(gt["bwd_flow"] if "bwd_flow" in gt else ch_sec(model_out["flow_inp"]),model_out["ctxt_rays"])
    losses["metrics/flow_from_pose"] = ((model_out["flow_from_pose"].clip(-.2,.2) - gt_bwd_flow.clip(-.2,.2)).square().mean() * 6e3).clip(0,10)

    gt_bwd_flow_trgt = rays(gt["bwd_flow"] if "bwd_flow" in gt else ch_sec(model_out["flow_inp"]),model_out["trgt_rays"])

    # monodepth loss (not used in paper but may be useful later)
    if args.midas_sup:
        def depth_lstsq_fit(depthgt,depthest):
            depthgt,depthest=1/(1e-8+depthgt),1/(1e-8+depthest)
            lstsq=torch.linalg.lstsq(depthgt,depthest)
            return ((depthgt@lstsq.solution)-depthest).square().mean() * 1e2

        losses["metrics/ctxt_depth_loss_lstsq"] = (depth_lstsq_fit(rays(ch_sec(model_out["ctxt_depth_inp"]),model_out["ctxt_rays"]).flatten(0,1),model_out["ctxt_depth"].flatten(0,1))*2e0).clip(0,2)/2
        losses["metrics/depth_loss_lstsq"] = (depth_lstsq_fit(rays(ch_sec(model_out["trgt_depth_inp"]),model_out["trgt_rays"]).flatten(0,1),model_out["depth"].flatten(0,1))*2e0).clip(0,2)/2

    return losses

model = torch.nn.DataParallel(model)

# Train loop
for step in trange(0 if args.eval else int(1e8), desc="Fitting"): # train until user interruption

    # Run val set every n iterations
    val_step = step>10 and step%150<10 
    prefix = "val" if val_step else ""
    torch.set_grad_enabled(not val_step)
    if val_step: print("\n\n\nval step\n\n\n")

    # Get data
    try: model_input, ground_truth = next(train_dataloader if not val_step else val_dataloader)
    except StopIteration:
        if val_step: val_dataloader = get_dataloader(val_dataset)
        else: train_dataloader = get_dataloader(train_dataset)
        continue

    model_input, ground_truth = to_gpu(model_input), to_gpu(ground_truth)

    # Run model and calculate losses
    total_loss = 0.
    for loss_name, loss in loss_fn(model(model_input), ground_truth, model_input).items():
        wandb.log({prefix+loss_name: loss.item()}, step=step)
        total_loss += loss

    wandb.log({prefix+"loss": total_loss.item()}, step=step)
    wandb.log({"epoch": (step*args.batch_size)/len(train_dataset)}, step=step)

    if not val_step: 
        optim.zero_grad(); total_loss.backward(); optim.step();

    # Image summaries and checkpoint
    if step%50==0 : # write image summaries
        print("writing summary")
        with torch.no_grad(): model_output = model.module.render_full_img(model_input=model_input)
        vis_scripts.wandb_summary( total_loss, model_output, model_input, ground_truth, None,prefix=prefix)
    if step%100==0: #write video summaries
        print("writing video summary")
        try: vis_scripts.write_video(save_dir, vis_scripts.render_time_interp(model_input,model.module,None,16), prefix+"time_interp",step)
        except Exception as e: print("error in writing video",e)
    if step%500 == 0 and step: # save model
        print(f"Saving to {save_dir}"); torch.save({ 'step': step, 'model_state_dict': model.module.state_dict(), }, os.path.join(save_dir, f"checkpoint_{step}.pt")) 


