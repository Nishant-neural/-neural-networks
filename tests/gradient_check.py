import numpy as np

def gradient_check(model, X, Y, loss_fn, epsilon=1e-5):

    preds = model.forward(X)
    loss = loss_fn.forward(preds, Y)
    grad = loss_fn.backward()
    model.backward(grad)

    print("Running gradient check...")

    for layer in model.layers:
        if not hasattr(layer, "W"):
            continue

        for param_name in ["W", "b"]:

            param = getattr(layer, param_name)
            grad_analytic = getattr(layer, "d" + param_name)

          
            idx = tuple(np.random.randint(s) for s in param.shape)

            original_value = param[idx]

           
            param[idx] = original_value + epsilon
            plus_loss = loss_fn.forward(model.forward(X), Y)

         
            param[idx] = original_value - epsilon
            minus_loss = loss_fn.forward(model.forward(X), Y)

           
            param[idx] = original_value

            grad_numeric = (plus_loss - minus_loss) / (2 * epsilon)

            print(f"{param_name} check:",
                  "Analytic:", grad_analytic[idx],
                  "Numeric:", grad_numeric)