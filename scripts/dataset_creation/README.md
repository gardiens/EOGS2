permet de créer les caméras affines du modèles.

# Ce qu'il renvoie :

On suppose ici que les données ont déjà été pré bundle adjuster . Ils ont été bundle adjust à partir de cette adresse
https://github.com/centreborelli/satnerf?tab=readme-ov-file#41-dataset-creation-from-the-dfc2019-data .

Les caméras non ajustés sont localisé dans le dossier rpcs_raw .


Il renvoie un fichier affine_model.json Dans lequel :

-> center of scene correspond à 0,0,0
model correspond au modèle affine avec les différents paramètres
'''
        "coef_": A.tolist(),
        "intercept_": b.tolist(),
        "scale": converter.scale,
        "n": converter.n,
        "l": converter.l,
        # "rotation": converter.R.tolist(),
        "center": converter.shift.tolist(),
        "min_world": converter.min_world.tolist(),
        "max_world": converter.max_world.tolist(),

'''

le model est celui là
'''

        "coef_": sun_A.tolist(),
        "intercept_": sun_b.tolist(),
        "sun_dir_ecef": sun_dir_ecef.tolist(),
        "camera_to_sun": myM.tolist(),
'''

les hyperparamètres sont ceux là
'''
        "img": "JAX_068_001_RGB.tif",
        "height": 815,
        "width": 746,
        "sun_elevation": "+50.8",
        "sun_azimuth": "150.3",
        "acquisition_date": "20141005160138",
        '''