from tabularbench.benchmark.benchmark import benchmark


def run():
    print("Welcome to TabularBench!")

    clean_acc, robust_acc = benchmark(
        dataset="URL",
        model="STG_madry",
        distance="L2",
        constraints=True,
    )
    print("LCLD on STG madry(adversary)")
    print(f"Clean accuracy: {clean_acc}")
    print(f"Robust accuracy: {robust_acc}")
        

if __name__ == "__main__":
    run()
