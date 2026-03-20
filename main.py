from src.model import train_model, predict

if __name__ == "__main__":
    print("Training model...")
    model = train_model()

    while True:
        path = input("\nEnter image path (or q to quit): ")

        if path.lower() == 'q':
            break

        result = predict(model, path)
        print("Result:", result)