from app.inference import WorldModelInference


def main():
    engine = WorldModelInference()

    print()
    print("State features:", len(engine.get_state_columns()))

    timestamp = engine.network_states.iloc[
        300
    ]["TimeWindow"]

    print("Testing timestamp:", timestamp)

    result = engine.analyze_timestamp(timestamp)

    print()
    print("Prediction MSE:", result.get("mse"))
    print("Prediction MAE:", result.get("mae"))

    print()
    print("INFERENCE TEST PASSED")


if __name__ == "__main__":
    main()