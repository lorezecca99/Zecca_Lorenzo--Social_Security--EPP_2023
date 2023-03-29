import numpy as np
import pytask

from ss_us.analysis.steady_states import SS

from ss_us.config import BLD, SRC
from ss_us.utilities import read_yaml

@pytask.mark.depends_on(
    {
        "scripts": ["steady_states.py"],
        "data": BLD / "python" /"age_efficiency"/ "eff_profiles_py.txt",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.produces(
    {BLD / "python" / "results"}
)
def task_SS_python(depends_on, produces):
    """Compute the ss and save data for figures."""
    #data_info = read_yaml(depends_on["data_info"])
    data = np.loadtxt(depends_on["data"])
    iterations=2 

    """In Matlab we usually need 10. 
    In python, the results seem reasonable if iterations>=30. It can be set also to, 
    let us say, 2 just to prove that the code is running and all the results are stored. Of course, final results, 
    and figures will be totally unreliable for the analysis."""

    ss_results = SS(data,iterations)
    """Saving all the scalar as .txt and vectors as .npy"""
    np.save(BLD / "python" / "results"/"kgen0.npy", ss_results[0])
    np.save(BLD / "python" / "results"/"kgen1.npy", ss_results[1])
    np.save(BLD / "python" / "results"/"labgen0.npy", ss_results[2])
    np.save(BLD / "python" / "results"/"labgen1.npy", ss_results[3])
    with open(BLD / "python" / "results"/'r0_0.txt', 'w') as f:
        f.write(str(ss_results[4]))
    with open(BLD / "python" / "results"/'w0_0.txt', 'w') as f:
        f.write(str(ss_results[5]))
    with open(BLD / "python" / "results"/'K_0.txt', 'w') as f:
        f.write(str(ss_results[6]))
    with open(BLD / "python" / "results"/'L_0.txt', 'w') as f:
        f.write(str(ss_results[7]))
    with open(BLD / "python" / "results"/'b_0.txt', 'w') as f:
        f.write(str(ss_results[8]))
    with open(BLD / "python" / "results"/'r0_1.txt', 'w') as f:
        f.write(str(ss_results[9]))
    with open(BLD / "python" / "results"/'w0_1.txt', 'w') as f:
        f.write(str(ss_results[10]))
    with open(BLD / "python" / "results"/'K_1.txt', 'w') as f:
        f.write(str(ss_results[11]))
    with open(BLD / "python" / "results"/'L_1.txt', 'w') as f:
        f.write(str(ss_results[12]))
    with open(BLD / "python" / "results"/'b_1.txt', 'w') as f:
        f.write(str(ss_results[13]))
    np.save(BLD / "python" / "results"/"earn_0.npy", ss_results[14])
    np.save(BLD / "python" / "results"/"earn_1.npy", ss_results[15])
    np.save(BLD / "python" / "results"/"cons_0.npy", ss_results[16])
    np.save(BLD / "python" / "results"/"cons_1.npy", ss_results[17])
    with open(BLD / "python" / "results"/'V0_0.txt', 'w') as f:
        f.write(str(ss_results[18]))
    with open(BLD / "python" / "results"/'V0_1.txt', 'w') as f:
        f.write(str(ss_results[19]))
    with open(BLD / "python" / "results"/'CEV.txt', 'w') as f:
        f.write(str(ss_results[20]))
    return produces

