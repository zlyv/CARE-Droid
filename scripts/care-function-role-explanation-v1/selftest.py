#!/usr/bin/env python3
from taxonomy import classify
def main():
    assert "UI_INTERACTION" in classify("Landroid/widget/Button;->setText(Ljava/lang/CharSequence;)V")
    assert "LOCATION_GEO" in classify("Landroid/location/LocationManager;->getProviders(Z)Ljava/util/List;")
    assert not classify("Lcom/example/Secret;->run()V")
    print("FCBE_SELFTEST_PASS static=true raw_identity_feature=false")
if __name__=="__main__":main()
