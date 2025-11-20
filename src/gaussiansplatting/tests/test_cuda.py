import torch


if __name__ == "__main__":
    # put a tensor of size 1000*1000*100 on the PGU and wait 20 sec
    x = torch.randn(1000, 1000, 100).cuda()
    print("Tensor allocated on GPU. Waiting for 20 seconds...")
    torch.cuda.synchronize()  # Ensure all operations are complete
    import time

    time.sleep(20)
    print("Done waiting.")
