from scipy.optimize import differential_evolution
import numpy as np
from keras import models


def perturbImage(image, label, model: models.Model):
    def getPerturbImage(perturb):
        new_image = image.copy()
        new_image[int(perturb[0])][int(perturb[1])] = (perturb[2], perturb[3], perturb[4])
        return new_image

    def getModelStat(perturb):
        new_image = getPerturbImage(perturb)
        predictions = model.predict(np.array([new_image]), )
        return predictions[0][label]

    bounds = [(0, 32), (0, 32), (0, 1), (0, 1), (0, 1)]
    result = differential_evolution(func=getModelStat, bounds=bounds, maxiter=75, popsize=80, recombination=1, atol=-1, polish=False)
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
    print(f'Pre attack scores: {pre_attack[0]}, class: {np.argmax(pre_attack[0])}')
    print(f'Pre attack scores: {post_attack[0]}, class: {np.argmax(post_attack[0])}')
