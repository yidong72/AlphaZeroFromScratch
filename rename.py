import fire


def remove_module(pt_file, out_file):
    import torch
    model = torch.load(pt_file)
    for key in list(model.keys()):
        if key.startswith('module.'):
            model[key[7:]] = model[key]
            del model[key]
    torch.save(model, out_file)
    print('Done')

if __name__ == '__main__':
    fire.Fire(remove_module)