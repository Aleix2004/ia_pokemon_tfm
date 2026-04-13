try:
    from src.train_ia import train
except ImportError:
    from train_ia import train


if __name__ == "__main__":
    train()
