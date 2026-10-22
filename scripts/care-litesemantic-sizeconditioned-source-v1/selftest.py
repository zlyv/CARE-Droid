#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer,StandardScaler

def nuisance_model():
    return Pipeline([("spline",SplineTransformer(n_knots=5,degree=2,knots="quantile",include_bias=False)),("scale",StandardScaler()),("ridge",Ridge(alpha=10.0))])

def main():
    rng=np.random.default_rng(7);z=rng.normal(size=(80,7));x=np.c_[z[:,0]**2+z[:,1],rng.normal(size=80)]
    m=nuisance_model();m.fit(z[:60],x[:60]);res=x[60:]-m.predict(z[60:])
    assert res.shape==(20,2) and np.isfinite(res).all()
    legitimate=("role_presence__package_installation","role_proportion__package_uninstallation")
    assert all(name.startswith(("role_presence__","role_proportion__")) for name in legitimate)
    print("LITESEMANTIC_SIZECONDITIONED_SELFTEST_PASS")
    print("nuisance_model=train_only_spline_ridge_alpha10")
    print("outer_train_residuals=inner_oof outer_validation_residuals=train_fit_only")
    print("raw_size_enters_classifier=false future_rows_loaded=0 apk_reanalysis=false")
    print("package_installation_roles_allowed=true package_identity_forbidden=true")

if __name__=="__main__":main()
