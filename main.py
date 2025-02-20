import argparse
import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "api"], default="train",
                        help="Run training or API server")
    args = parser.parse_args()
    
    if args.mode == "train":
        train.main()
    elif args.mode == "api":
        import uvicorn
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()