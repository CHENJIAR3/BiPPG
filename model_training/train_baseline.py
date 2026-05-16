
from utils import args
from trainers.ppg_trainer_simple import PPG_trainer_twohands

from dataloaders.dataloading import get_dataloader
from evaluation.model_metric import save_all_metrics_to_xlsx,get_performance
import datetime
formatted_date = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M-%S")
from models.model_loading import load_model

def run_exp(trainer,data_dir,args,dataname=None):

    if args.trainflag:
        if dataname is None:
            train_loader = get_dataloader(data_dir, args, split=f'train')
            val_loader = get_dataloader(data_dir, args, split=f'val')
        else:
            train_loader = get_dataloader(data_dir, args, split=f'{dataname}_train')
            val_loader = get_dataloader(data_dir, args, split=f'{dataname}_val')
        trainer.train(train_loader,val_loader)

    if args.testflag:
        if dataname is None:
            test_loader = get_dataloader(data_dir, args, split=f'test')
        else:
            test_loader = get_dataloader(data_dir, args, split=f'{dataname}_test')
        predictions, targets = trainer.predict(test_loader)
        return predictions, targets
    else:
        return trainer


def get_results(predictions,targets,hand_idx=0,pred_num=4):
    predictions_mean = 0.5* (predictions[:, :pred_num] +predictions[:, pred_num:])
    result = get_performance(predictions_mean, targets)
    return result

if __name__ == "__main__":
    data_dir = "/home/cjr/datasets/Ring2Health/dataset/"
    args.norm_method = "z-score"    # "z-score","maxmin","rms"
    print(args.norm_method)
    args.device = "cuda:2"
    args.index = range(1,16,2)

    args.in_channels = len(args.index)//2
    # args.epochs = 1
    dataname = "all"
    # [0,1,4,5],range(8)self.index = [0,1,2,3,8,9,10,11,
    #                       4,5,6,7,12,13,14,15]
    all_results = {}

    result_file = f"../results/performance{formatted_date}.xlsx"
    for model_type  in ["Net1D",]:
        args.model_type = model_type
        subgroup = args.exp +"_" + args.model_type
        args.modelpath = f"../saved_models/PPG2BP_{subgroup}_{formatted_date}"
        model = load_model(args)
        trainer =  PPG_trainer_twohands(model,args)
        # args.trainflag = False
        predictions, targets  = run_exp(trainer,data_dir,args,dataname=dataname)
        all_results[subgroup] = get_results(predictions,targets)

    save_all_metrics_to_xlsx(all_results,result_file)
