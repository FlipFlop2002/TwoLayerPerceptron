from models import *
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

results = {}
for hu in range(6, 7):
    # parameters
    HU = hu
    print(HU)
    LR = 0.01
    EPOCHS = 12000
    print(hu)


    X, Y = fx(0, 1, -8, 8, 10000)
    perceptron = TwoLayerPerceptron(input_size=1, hidden_units=HU, output_size=1, seed=42)
    preds = perceptron.predict(X)
    # plt.plot(X, Y)
    # plt.plot(X, preds)
    # plt.show()
    # plt.savefig('wykres.png')

    data = np.array([X, Y])

    # podział danych na traningowe, weryfikacyjne i testowe
    X_train, Y_train, X_eval, Y_eval, X_test, Y_test = train_eval_test_split(X, Y)
    print(X_train.shape, Y_train.shape, X_eval.shape, Y_eval.shape, X_test.shape, Y_test.shape)


    # training dataset
    training_dataset = create_dataset(X_train, Y_train)
    eval_dataset = create_dataset(X_eval, Y_eval)
    test_dataset = create_dataset(X_test, Y_test)



    preds = []
    print('TRAINING')
    eval_mse_per_epoch = []
    eval_mae_per_epoch = []
    # training
    for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):
        y_pred_eval = perceptron.predict(X_eval)
        eval_mse_per_epoch.append(calc_mse(Y_eval, y_pred_eval))
        eval_mae_per_epoch.append(calc_mae(Y_eval, y_pred_eval))
        preds = []
        for x, y in training_dataset:
            pred = perceptron.forward(x)
            perceptron.backward(y, pred, LR)

    plt.plot(eval_mse_per_epoch)
    plt.title("Zbiór weryfikujący")
    plt.xlabel("epoka")
    plt.ylabel("MSE")
    # plt.savefig('6_12000_mse.png')
    plt.show()

    plt.plot(eval_mae_per_epoch)
    plt.title("Zbiór weryfikujący")
    plt.xlabel("epoka")
    plt.ylabel("MAE")
    # plt.savefig('6_12000_mae.png')
    plt.show()

    # test
    y_pred_test = perceptron.predict(X_test)
    print(f'TEST MSE:{round(calc_mse(Y_test, y_pred_test), 6)}')
    print(f'TEST MAE:{round(calc_mae(Y_test, y_pred_test), 6)}')
    mse = round(calc_mse(Y_test, y_pred_test), 6)
    mae = round(calc_mae(Y_test, y_pred_test), 6)
    results[hu] = [mse, mae]
    plt.plot(X_test, Y_test,label="f(x)")
    plt.plot(X_test, y_pred_test, label="predykcja")
    plt.legend()
    plt.title("Zbiór testowy")
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.savefig('6_12000_aproks.png')
    plt.show()

# print(results)
# sciezka_pliku = "wyniki.json"

# # Zapisywanie słownika do pliku JSON
# with open(sciezka_pliku, "w") as plik:
#     json.dump(results, plik, indent=4)