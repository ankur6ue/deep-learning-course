import torch, time, gc

# Timing utilities
start_time = None

# src: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    max_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Max memory used by tensors = {max_gb:.2f} GB")


def make_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()


batch_size = 1024 # Try, for example, 128, 256, 513.
in_size = 4096
out_size = 4096
num_layers = 9
num_batches = 50
epochs = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

# Creates data in default precision.
# The same data is used for both default and mixed precision trials below.
# You don't need to manually change inputs' ``dtype`` when enabling mixed precision.
data = [torch.randn(batch_size, in_size) for _ in range(num_batches)]
targets = [torch.randn(batch_size, out_size) for _ in range(num_batches)]

loss_fn = torch.nn.MSELoss().cuda()

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        output = net(input)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
end_timer_and_print("Default precision:")

use_amp = True

# net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.amp.GradScaler("cuda" ,enabled=use_amp)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
end_timer_and_print("Mixed precision:")