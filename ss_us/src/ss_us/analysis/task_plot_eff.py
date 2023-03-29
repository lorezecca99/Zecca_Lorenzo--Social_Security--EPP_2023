
import pandas as pd
import pytask
import matplotlib.pyplot as plt
import os

from ss_us.analysis.predict import predict_eff_age
from ss_us.config import BLD, SRC
from ss_us.utilities import read_yaml


figure_dir = BLD / "python" / "age_efficiency" / "figure"

if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)             ##this is needed, otherwise, the folder is not created by "produces"
    
@pytask.mark.depends_on(
    {
        "scripts": ["predict.py"],
        "data": BLD / "python" / "age_efficiency" / "age_eff.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.produces(
    {BLD / "python" / "age_efficiency"/"figure"}
)
def task_age_eff_txt_python(depends_on, produces):
    """Figure (Python version)."""
    data=pd.read_csv(depends_on["data"])
    fig_eff=predict_eff_age(data)
    fig_eff.savefig(BLD / "python" / "age_efficiency" / "figure"/ "eff_age_profile.png")
    #fig_eff.savefig(BLD/"python"/ "age_efficiency"/ "figure"/"eff_age_profile.png")
    return produces