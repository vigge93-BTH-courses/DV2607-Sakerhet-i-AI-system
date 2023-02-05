from scipy.optimize import differential_evolution
import numpy as np
from data import loadModel
from keras import models


def perturbImage(image, label, model: models.Model):
    i = 1
    def getPerturbImage(perturb):
        new_image = image.copy()
        new_image[int(perturb[0])][int(perturb[1])] = (perturb[2], perturb[3], perturb[4])
        return new_image

    getPerturbImage_vector = np.vectorize(getPerturbImage, signature='(n)->(k,k,m)')

    def getModelStatVector(perturb):
        new_images = getPerturbImage_vector(perturb.T)
        predictions = model.predict(new_images, verbose=0)
        return predictions[:, label]

    def callback(xk, convergence):
        nonlocal i
        new_image = getPerturbImage(xk)
        res = model.predict(np.array([new_image]), verbose=0)[0]
        if i % 10 == 0 or res[label] <= 0.05:
            print(f'Best results so far: {res[label]*100:.1f}%')
        i += 1
        return res[label] <= 0.05
        predicted = np.argmax(res)
        return predicted != label

    bounds = [(0, 32), (0, 32), (0, 1), (0, 1), (0, 1)]
    result = differential_evolution(func=getModelStatVector, callback=callback, bounds=bounds, maxiter=100, recombination=1, atol=-1, popsize=80, polish=False, disp=False, vectorized=True)

    return getPerturbImage(result.x)


if __name__ == '__main__':
    from data import getCifar10
    import matplotlib.pyplot as plt
    from NiN import getModelNiN, fit
    x_train, y_train, x_test, y_test = getCifar10()
    img = x_test[1]
    print(y_test[1])
    plt.imshow(img)
    plt.show()
    model = getModelNiN()
    # fit(model, x_train, y_train, x_test, y_test)
    # saveModel(model, 'NiN_trained.model')
    pre_attack = model.predict(np.array([img]))
    attacked = perturbImage(img, 8, model)

    post_attack = model.predict(np.array([attacked]))
    plt.imshow(attacked)
    plt.show()
    print(f'Pre attack scores: {pre_attack[0].round(3)}, class: {np.argmax(pre_attack[0])}')
    print(f'Pre attack scores: {post_attack[0].round(3)}, class: {np.argmax(post_attack[0])}')
