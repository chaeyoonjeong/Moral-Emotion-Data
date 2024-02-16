import numpy as np
import pandas as pd
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import random
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader #, TensorDataset
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import argparse
from imblearn.over_sampling import RandomOverSampler
from ENG_modules.datasets import *
from ENG_modules.models import EnglishMultiMoralEmotionTagger, EnglishSingleMoralEmotionTagger
from datetime import datetime


def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def oversample_dataframe(df, target_column, least_class="Self-Conscious", second_least_class="Other-Praising"):
    class_counts = df[target_column].value_counts()
    desired_samples = class_counts[second_least_class]
    oversampler = RandomOverSampler(sampling_strategy={least_class: desired_samples})
    X, y = oversampler.fit_resample(df.drop(columns=[target_column]), df[target_column])
    df_oversampled = pd.concat([pd.DataFrame(X), pd.DataFrame({target_column: y})], axis=1)

    return df_oversampled


def prepare_dataset(args, tokenizer):
    print("==> Prepare Datasets")
    eng_df = pd.read_parquet("./petition_50k_test/ENG_TrainingSet_0924.parquet")

    if args.use_single_label:
        eng_df = eng_df[eng_df["fine_tuning_label"]=="Single"].reset_index(drop=True)

    # pl.seed_everything(args.random_seed)  # 이 부분은 필요한 라이브러리가 언급되지 않아서 주석 처리하였습니다.
    df_train, df_test = train_test_split(eng_df, train_size=.8, random_state=args.random_seed)
    df_val, df_test = train_test_split(df_test, train_size=.25, random_state=args.random_seed)
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    df_train =df_train.rename(columns={'sentence_token':'text', 'fine_tuning_predict':'labels'})
    df_test =df_test.rename(columns={'sentence_token':'text', 'fine_tuning_predict':'labels'})
    df_val =df_val.rename(columns={'sentence_token':'text', 'fine_tuning_predict':'labels'})
    df_train = oversample_dataframe(df_train, 'labels')
    print(f"Training set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")


    
    mlb = MultiLabelBinarizer(classes=args.labels)

    if not args.use_single_label:
        df_train["labels"] = df_train["labels"].str.split(",")
        df_test["labels"] = df_test["labels"].str.split(",")
        df_val["labels"] = df_val["labels"].str.split(",")
        train_labels = mlb.fit_transform(df_train["labels"])
        test_labels = mlb.fit_transform(df_test["labels"])
        val_labels = mlb.fit_transform(df_val["labels"])
    else:
        train_labels = mlb.fit_transform(df_train["labels"].apply(lambda x:[x]))
        test_labels = mlb.fit_transform(df_test["labels"].apply(lambda x:[x]))
        val_labels = mlb.fit_transform(df_val["labels"].apply(lambda x:[x]))


    train_texts = df_train['text']
    test_texts = df_test['text']
    val_texts = df_val['text']

    train_dataset = EnglishMoralEmotionDataset(train_texts, train_labels, tokenizer=tokenizer, augmentation=args.aug)
    test_dataset = EnglishMoralEmotionDataset(test_texts, test_labels, tokenizer=tokenizer)
    val_dataset = EnglishMoralEmotionDataset(val_texts, val_labels, tokenizer=tokenizer)
    
    data_module = EnglishMoralEmotionDataModule(train_dataset, test_dataset, val_dataset, batch_size=args.batch_size, )
    
    return data_module

def prepare_human_dataset(args, tokenizer):
    print("==> Prepare Human labeled dataset")
    test_df = pd.concat([pd.read_csv("/home/kjhkjh95/국민청원/Training_Set_Sampling/300_300_set/ENG_train_set.csv"), 
                             pd.read_csv("/home/kjhkjh95/국민청원/Training_Set_Sampling/300_300_set/ENG_test_set.csv"),], axis=0).reset_index(drop=True)
    test_df = test_df.rename(columns={'Question':'text'})
    mlb = MultiLabelBinarizer(classes=args.labels)
    test_df["labels"] = test_df['Gold Label'].str.split(',')
    test_df_labels = mlb.fit_transform(test_df["labels"])
    test_dataset = EnglishMoralEmotionDataset(texts=test_df["text"], labels=test_df_labels, tokenizer=tokenizer)
    return test_dataset
    
def predict_petition_dataset(args, tokenizer, model, output_dir = "./output_petition/"):
    print("==> Prepare Petition dataframe")
    # petition_df = pd.read_parquet("/home2/emotion/petition_50k_test/Prediction_Target_Sentences/ENG_prediction_target.parquet").reset_index(drop=True)
    petition_df = pd.read_parquet("/home2/emotion/petition_50k_test/Prediction_Target_Sentences/ENG_prediction_target_no_threshold.parquet").reset_index(drop=True)
    petition_dataset = EnglishMoralEmotionDataset_NoLabel(texts=petition_df["sentence_token"], tokenizer=tokenizer)

    petition_dataloader = DataLoader(petition_dataset, batch_size=args.test_batch_size) 

    model.eval()
    print("Start at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    predictions = []
    for iter, item in enumerate(petition_dataloader, start=1):
        _, pred = model(
            item["input_ids"].to(args.main_device),
            item["attention_mask"].to(args.main_device)
        )
        predictions.append(pred.detach().cpu().numpy())
        del item
        torch.cuda.empty_cache()
        if len(predictions) >= len(petition_dataset):
            break
        if iter % 200 == 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: [{iter}]/[{len(petition_dataloader)}]")
        
    predictions = np.concatenate(predictions, axis=0)
    
    
    os.makedirs(output_dir, exist_ok=True)
    label_type = "Single" if args.use_single_label else "Multi"
    model_name_save = args.model_name.replace("/", "-")
    savedir = os.path.join(output_dir, f"ENG_{label_type}_label_{model_name_save}_{args.postfix}.npy")
    args.output_save_dir = savedir
    print("==> Save Results at", savedir)
    np.save(savedir, predictions)
    

def load_model_tokenizer(args):
    print("==> Load models")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder_model = AutoModel.from_pretrained(args.model_name, return_dict=True)
    
    return encoder_model, tokenizer


def train(args, model, datamodule):
    label_type = "Single" if args.use_single_label else "Multi"
    model_name_save = args.model_name.replace("/", "-")
    savedir = f"./model_ckpt/ENG_{label_type}_label_{model_name_save}_{args.postfix}/"
    args.model_save_dir = savedir
    os.makedirs(savedir, exist_ok=True)
    print("==> Model saved at", savedir)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=savedir,
        filename="ENG_epoch{epoch}-val_loss_{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=False,
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2, min_delta=0.00)

    logger = TensorBoardLogger("emotion_model", name="logger")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=args.n_epochs,
        accelerator="auto",
        devices=args.gpus,
        strategy="ddp_notebook",
        enable_progress_bar=False
    )

    trainer.fit(model, datamodule=datamodule)
    
    return trainer, model

def predict(args, model, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size) 

    model.eval()

    predictions = []
    labels = []
    for item in test_dataloader:
        _, pred = model(
            item["input_ids"].to(args.main_device),
            item["attention_mask"].to(args.main_device)
        )
        predictions.append(pred.detach().cpu().numpy())
        labels.append(item["labels"].round().int().detach().cpu().numpy())
        del item
        torch.cuda.empty_cache()
        if len(predictions) >= len(test_dataset):
            break
        
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    return labels, predictions


def evaluate(args, model, test_dataset, output_dir="./output/", threshold=[0.4]):
    print("==> Evaluate")
    labels, predictions = predict(args, model, test_dataset)
    
    os.makedirs(output_dir, exist_ok=True)
    label_type = "Single" if args.use_single_label else "Multi"
    model_name_save = args.model_name.replace("/", "-")
    savedir = os.path.join(output_dir, f"ENG_{label_type}_label_{model_name_save}_{args.postfix}.npy")
    args.output_save_dir = savedir
    print("==> Save Results at", savedir)
    
    np.save(os.path.join(output_dir, "gt.npy"), labels)
    np.save(savedir, predictions)
    
    if args.use_single_label:
        labels = np.argmax(labels, axis=1)
        predictions = np.argmax(predictions, axis=1)
        r = classification_report(labels, predictions, target_names=args.labels, digits=4)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        print(r)
        print(acc)
        return {"classification_report":r, "accuracy_score": acc}
        
    else:
        result_df = dict()
        reports = []
        accuracies = []
        f1s = []
        for t in threshold:
            print("Threshold :", t)
            predicted_label = (predictions > t).astype(int)
            print("Zero labels:", np.sum(np.all(predicted_label == 0, axis=1)))
            r = classification_report(labels, predicted_label, target_names=args.labels, digits=4)
            acc = accuracy_score(labels, predicted_label)
            f1 = f1_score(labels, predicted_label, average='macro')
            
            reports.append(r)
            accuracies.append(acc)
            f1s.append(f1)

            print(r)
            print("accuracy :", acc)
            print("macro_f1 :", f1)
        result_df["threshold"] = threshold
        result_df["macro f1"] = f1s
        result_df["accuracy"] = accuracies
        result_df = pd.DataFrame.from_dict(result_df)
        df_save_dir = os.path.join(output_dir, "result_dfs/")
        os.makedirs(df_save_dir, exist_ok=True)
        result_df.to_csv(os.path.join(df_save_dir, f"ENG_{label_type}_label_{model_name_save}_{args.postfix}.csv"), index=False)
        return {"classification_report": reports, "accuracy_score": accuracies}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="English petition classification")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the encoder model")
    parser.add_argument("--postfix", type=str, default="", help="Postfix string")
    parser.add_argument("--gpus", type=str, default="0,1")
    
    parser.add_argument("--use_single_label", action="store_true", help="Single/Multi label")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of finetuning epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--test_batch_size", type=int, default=64, help="Batch size when inference")
    parser.add_argument("--max_length", type=int, default=128, help="Max length for LM")
    parser.add_argument("--initial_lr", type=float, default=2e-5, help="initial learning rate")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed number")
    parser.add_argument("--threshold", type=float, nargs='+', default=[0.2, 0.3, 0.35, 0.4, 0.5, 0.6])
    parser.add_argument("--aug", action="store_true", help="token_mask_and_switch")
    
    parser.add_argument("--checkpoint", type=str, help="location of checkpoint to load")
    parser.add_argument("--predict_petition", action="store_true", help="predict on petition data")
    parser.add_argument("--predict_human_labeled", action="store_true", help="predict on human labeled goldset")
    parser.add_argument("--predict_testset", action="store_true", help="predict on test set")
    parser.add_argument("--skip_train", action="store_true", help="skip training")
    
    
    args = parser.parse_args()
    seed_everything(args.random_seed)
    args.gpus = list(map(int, args.gpus.split(",")))
    args.main_device = torch.device("cuda:"+str(args.gpus[0]) if torch.cuda.is_available() else "cpu")
    encoder_model, tokenizer = load_model_tokenizer(args)
    args.labels = ['Other-Condemning', 'Other-Praising', 'Other-Suffering', 'Self-Conscious','Others', 'Neutral']
    
    TOTAL_STEPS, WARMUP_STEPS = 0, 0
    if not args.skip_train or args.predict_testset:
        datamodule = prepare_dataset(args, tokenizer)

        steps_per_epoch = len(datamodule.train_dataset) // args.batch_size
        TOTAL_STEPS = steps_per_epoch * args.n_epochs
        WARMUP_STEPS = TOTAL_STEPS // 5

    if args.use_single_label:
        model = EnglishSingleMoralEmotionTagger(encoder_model, args=args, n_warmup_steps=WARMUP_STEPS,n_training_steps=TOTAL_STEPS,)
    else:
        model = EnglishMultiMoralEmotionTagger(encoder_model, args=args, n_warmup_steps=WARMUP_STEPS,n_training_steps=TOTAL_STEPS,)
        
    if args.checkpoint:
        model = model.load_from_checkpoint(args.checkpoint, encoder_model=encoder_model, args=args).to(args.main_device)
    else:
        trainer, model = train(args, model, datamodule)
        
        best_ckpt = glob(args.model_save_dir+"*")[0] # <--
        
        model = model.load_from_checkpoint(best_ckpt, encoder_model=encoder_model, args=args).to(args.main_device)
    
    if args.predict_testset:
        eval_result = evaluate(args, model, datamodule.test_dataset, threshold=args.threshold)
        
    if args.predict_human_labeled:
        print("==> Evaluate on human goldset")
        test_dataset = prepare_human_dataset(args, tokenizer)
        eval_result = evaluate(args, model, test_dataset, output_dir="./output_human_labeled/", threshold=args.threshold)
    
    if args.predict_petition:
        print("==> Predict Petition Sentences")
        petition_dataset = predict_petition_dataset(args, tokenizer, model, output_dir="./output_petition/")
        