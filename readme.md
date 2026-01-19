



# MARL System Runner

This project provides a framework for running Multi-Agent Reinforcement Learning (MARL) experiments with customizable profiles and session tracking.

## 📦 Installing Dependencies

Make sure you have **Python 3.8+** installed. Install all required Python packages using:

```bash
pip install -r requirements.txt
```

## 📌 Example Use Case

You can run the main experiment script with a specific configuration using the following command:

```bash
python main.py --profile=github-marl-3h-qtran-5agent --session=1
```

This command will run WebCQ (qtran) on github for 3h.

| Argument    | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| `--profile` | Specifies the experiment configuration file (located in `./settings.yaml`). |
| `--session` | Custom session name to separate logs and results.            |

For each algorithm, we provide a example configuration profile name to help you get started quickly:

| **Algorithm** | Agent                   | Algo_type | **Example Profile**              |
| ------------- | ----------------------- | --------- | -------------------------------- |
| MARG_D        | multi_agent.impl.marg   | dql       | `github-marl-3h-marg-dql-5agent` |
| IDQN          | multi_agent.impl.marg_d | nn        | `github-marl-3h-nn-5agent`       |
| MARG_DQN      | multi_agent.impl.marg_d | nndql     | `github-marl-3h-nndql-5agent`    |
| WebCQ         | multi_agent.impl.marg_d | qtran     | `github-marl-3h-qtran-5agent`    |

## 🧠 Notes

Ensure the profile specified by `--profile` exists in the `settings.yaml` .

Logs and results are stored under `default_output_path`, defined in `./settings.yaml`.

Run multiple experiments by changing either the profile or the session name.

You can find our data split into 8 RAR archive parts in `./webtest_output/`.

## 🧪 Experimental Data

Due to GitHub's file size constraints, we have to upload the experimental data only on Zenodo. You can find the data in the folder `/webtest_output` at the following link: https://zenodo.org/records/17101249
