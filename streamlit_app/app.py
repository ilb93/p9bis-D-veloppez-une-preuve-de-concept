lightgbm.basic.LightGBMError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/p9bis-d-veloppez-une-preuve-de-concept/streamlit_app/app.py", line 148, in <module>
    proba = float(model.predict_proba(input_df)[0][1])
                  ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/lightgbm/sklearn.py", line 1627, in predict_proba
    result = super().predict(
        X=X,
    ...<6 lines>...
        **kwargs,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/lightgbm/sklearn.py", line 1144, in predict
    return self._Booster.predict(  # type: ignore[union-attr]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        X,
        ^^
    ...<6 lines>...
        **predict_params,
        ^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/lightgbm/basic.py", line 4767, in predict
    return predictor.predict(
           ~~~~~~~~~~~~~~~~~^
        data=data,
        ^^^^^^^^^^
    ...<6 lines>...
        validate_features=validate_features,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/lightgbm/basic.py", line 1204, in predict
    preds, nrow = self.__pred_for_np2d(
                  ~~~~~~~~~~~~~~~~~~~~^
        mat=data,
        ^^^^^^^^^
    ...<2 lines>...
        predict_type=predict_type,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/lightgbm/basic.py", line 1361, in __pred_for_np2d
    return self.__inner_predict_np2d(
           ~~~~~~~~~~~~~~~~~~~~~~~~~^
        mat=mat,
        ^^^^^^^^
    ...<3 lines>...
        preds=None,
        ^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/lightgbm/basic.py", line 1307, in __inner_predict_np2d
    _safe_call(
    ~~~~~~~~~~^
        _LIB.LGBM_BoosterPredictForMat(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<12 lines>...
        )
        ^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/lightgbm/basic.py", line 313, in _safe_call
    raise LightGBMError(_LIB.LGBM_GetLastError().decode("utf-8"))
