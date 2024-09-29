import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

sys.path.append("../")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))

# Initialize logger
_logger: logging.Logger = logging.getLogger(__name__)

from fast_jtnn import JTpropVAE, MolTreeFolder_prop, Vocab


def main_vae_train(
    args,
):
    #  Load the vocab
    with open(args.vocab) as f:
        vocab = f.read().splitlines()
    # vocab = [x.strip("\r\n ") for x in open(args.vocab)]
    vocab = Vocab(vocab)

    output_dir = Path(f"train_{time.strftime('%Y%m%d-%H%M%S')}")
    output_dir.mkdir(exist_ok=True)
    args.save_dir.mkdir(exist_ok=True)

    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "printlog.txt"), mode="w"),
            logging.StreamHandler(),  # For debugging. Can be removed on remote
        ],
    )
    beta = args.beta

    n_props = 2
    model = JTpropVAE(
        vocab,
        args.hidden_size,
        args.latent_size,
        args.depthT,
        args.depthG,
        n_props=n_props,
    ).cuda()
    print(model)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    if args.load_previous_model:
        model.load_state_dict(torch.load(args.model_path))

    print(
        (
            "Model #Params: %dK"
            % (sum([x.nelement() for x in model.parameters()]) / 1000,)
        )
    )

    # Write commandline args to file
    with open(output_dir / "opts.txt", "w") as file:
        file.write(f"{vars(args)}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    scheduler.step()

    def param_norm(m):
        return math.sqrt(sum([(p.norm().item() ** 2) for p in m.parameters()]))

    def grad_norm(m):
        return math.sqrt(
            sum(
                [
                    (p.grad.norm().item() ** 2)
                    for p in m.parameters()
                    if p.grad is not None
                ]
            )
        )

    total_step = args.load_epoch

    for epoch in tqdm(list(range(args.epoch)), position=0, leave=True):
        loader = MolTreeFolder_prop(
            args.train, vocab, args.batch_size, shuffle=False
        )  # , num_workers=4)
        for batch in loader:
            total_step += 1
            try:
                model.zero_grad()
                (
                    loss,
                    kl_div,
                    wacc,
                    tacc,
                    sacc,
                    prop_loss,
                    dent_loss,
                    isomer_loss,
                ) = model(batch, args.beta)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
            except Exception as e:
                print(e)
                continue

            meters = np.array(
                [
                    kl_div,
                    wacc * 100,
                    tacc * 100,
                    sacc * 100,
                    prop_loss * 100,
                    dent_loss * 100,
                    isomer_loss * 100,
                ]
            )

            if total_step % args.print_iter == 0:
                meters /= args.print_iter
                _logger.info(
                    (
                        "[%d] Loss: %.3f,Beta: %.3f,KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f,Prop_loss %.2f, Dent_loss %.2f, isomer_loss %.2f,PNorm: %.2f, GNorm: %.2f"
                        % (
                            total_step,
                            loss.item(),
                            beta,
                            meters[0],
                            meters[1],
                            meters[2],
                            meters[3],
                            meters[4],
                            meters[5],
                            meters[6],
                            param_norm(model),
                            grad_norm(model),
                        )
                    )
                )
                sys.stdout.flush()

            if total_step % args.save_iter == 0:
                torch.save(model.state_dict(), output_dir / f"model.iter-{total_step}")

            if total_step % args.anneal_iter == 0:
                scheduler.step()
                _logger.info(("learning rate: %.6f" % scheduler.get_lr()[0]))

            # Update the beta value
            if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
                beta = min(args.max_beta, beta + args.step_beta)

    torch.save(model.state_dict(), output_dir / f"model.epoch-{epoch}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--save_dir", required=True, type=Path)
    parser.add_argument("--load_previous_model", action="store_true")
    parser.add_argument("--model_path", required=False, type=Path)
    parser.add_argument("--load_epoch", type=int, default=0)

    parser.add_argument("--hidden_size", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=8)  # 32)
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--depthT", type=int, default=20)
    parser.add_argument("--depthG", type=int, default=3)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_norm", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=0.006)
    parser.add_argument("--step_beta", type=float, default=0.002)
    parser.add_argument("--max_beta", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=500)

    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--anneal_rate", type=float, default=0.9)
    parser.add_argument("--anneal_iter", type=int, default=1000)
    parser.add_argument("--kl_anneal_iter", type=int, default=3000)
    parser.add_argument("--print_iter", type=int, default=50)
    parser.add_argument("--save_iter", type=int, default=5000)

    args = parser.parse_args()
    print(args)

    main_vae_train(
        args=args,
    )
