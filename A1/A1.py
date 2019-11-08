from MODEL import RegressionModel
from DATA import UnlimitedDataWorks
import pandas as pd

columns = ["junk", "lat", "lon", "alt"]
raw_df = pd.read_csv("3D_spatial_network.txt", sep=',', header=None,
                     names=columns).drop("junk", 1).sample(frac=1)

pre_processor = UnlimitedDataWorks(deg=1)
X_train, Y_train, x_val, y_val, x_test, y_test = pre_processor.train_test_split(raw_df)

model = RegressionModel(N=pre_processor.count
                        X=X_train,
                        Y=Y_train,
                        x=x_test,
                        y=y_test,
                        xval=x_val,
                        yval=y_val)

model.fit()
# model.gradient_descent()
model.stocastic_gradient_descent(50250)
# model.gradient_descent_L1_reg()
# model.gradient_descent_L2_reg()

