from pathlib import Path

def get_config():
    return {
        "batch_size" : 8,
        "num_epochs" : 20,
        "lr" : 10**-4,
        "seq_len" : 350,
        "d_model" : 512,
        "datasource" : "opus_books",
        "lang_src" : "en",
        "lang_tgt" : "it",
        "model_folder" : "weights",
        "model_basename" : "tmodel_",
        "preload" : "latest",
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name" : "runs/tmodel"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weight_file_path(config):
    model_folder =  f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"

    weights_files = list(Path(model_folder).glob(model_folder))

    if len(weights_files) == 0:
        return None
    
    def get_epoch_number(filename):
        name = filename.name # debug it
        epoch_str = name[len(config['model_basename']):].split('.')[0]
        try:
            return int(epoch_str)
        except ValueError:
            return -1
        
    weights_files.sort(key=get_epoch_number)
    return str(weights_files[-1])
