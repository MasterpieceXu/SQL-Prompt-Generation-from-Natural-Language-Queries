"""Member 1: dataset loading, analysis, cleaning, split, and dataloader code."""

import pandas as pd


from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
from src.config import (RANDOM_SEED, 
                        DATASET_PATH, 
                        MAX_INPUT_LENGTH, 
                        MAX_TARGET_LENGTH,
                        BASELINE_MODEL_NAME)


def load_raw_dataset():
    df = pd.read_csv(DATASET_PATH)
    return df
    

def inspect_dataset(df):
    """Inspect dataset size, columns, missing values, duplicates, and samples."""
    # check the dataset
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    empty_prompt = (df["prompt"].astype(str).str.strip() == "").sum()
    empty_response = (df["response"].astype(str).str.strip() == "").sum()
    print("\nEmpty strings:")
    print("prompt:", empty_prompt)
    print("response:", empty_response)
    print("\nDuplicate rows:")
    print(df.duplicated().sum())


def clean_dataset(df):
    """Clean prompt/response pairs and return the cleaned dataset."""
    df_clean = df.copy()
    # clean the duplicated rows
    df_clean = df_clean.drop_duplicates().reset_index(drop = True)
    df_clean["prompt"]= df_clean["prompt"].str.strip()
    
    # get the pure sql from SQL
    df_clean["sql"] = (
        df_clean["response"]
        .str.replace("```sql", "", regex=False)
        .str.replace("```", "", regex=False)
        .str.strip()
    )
    return df_clean[["prompt", "sql"]]


def split_dataset(df):
    """Create train, validation, and test splits."""
    # First Split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
    # Second Split
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df,val_df,test_df

def tokenize_dataset(train_df, val_df, test_df, tokenizer):
    """Tokenize prompt and response fields for model training."""
    dataset = DatasetDict({"train": Dataset.from_pandas(train_df, preserve_index=False),
                           "validation": Dataset.from_pandas(val_df, preserve_index=False),
                           "test": Dataset.from_pandas(test_df, preserve_index=False)})

    def preprocess_function(examples):
        # prompt is the input of the model
        model_inputs = tokenizer(examples["prompt"], max_length=MAX_INPUT_LENGTH, truncation=True)
        # sql is the target output of the model
        labels = tokenizer(text_target = examples["sql"], max_length=MAX_TARGET_LENGTH, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["prompt","sql"])
    
    return tokenized_dataset


def build_dataloaders(tokenized_dataset, tokenizer, batch_size=8):
    """Build train, validation, and test DataLoaders."""
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")
    train_loader = DataLoader(tokenized_dataset["train"], batch_size=batch_size, shuffle=True, 
                                 collate_fn=data_collator,
                                 num_workers=0)
    val_loader = DataLoader(tokenized_dataset["validation"], batch_size=batch_size, shuffle=False, 
                                 collate_fn=data_collator,
                                 num_workers=0)
    test_loader = DataLoader(tokenized_dataset["test"], batch_size=batch_size, shuffle=False, 
                                 collate_fn=data_collator,
                                 num_workers=0)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    df = load_raw_dataset()
    inspect_dataset(df)
    df_clean = clean_dataset(df)
    train_df, val_df, test_df = split_dataset(df_clean)
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME)

    tokenized_dataset = tokenize_dataset(
        train_df,
        val_df,
        test_df,
        tokenizer
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        tokenized_dataset,
        tokenizer,
        batch_size=8
    )
    batch = next(iter(train_loader))

    print("\nBatch keys:", batch.keys())
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Attention mask shape:", batch["attention_mask"].shape)
    print("Labels shape:", batch["labels"].shape)
    print("\nTokenized dataset:")
    print(tokenized_dataset)

    print("\nDataLoader sizes:")
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))
    