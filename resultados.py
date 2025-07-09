import os
import pandas as pd
import numpy as np

# Ruta a la carpeta con los resultados
results_dir = "/Users/samucerezo/Desktop/init_bg2/results/from_bg_estimated"

# Lista para guardar resultados por secuencia
results = []

# "frame,error_small,elapsed_small_us,error_wo_preint,elapsed_wo_preint_us,error_opt,elapsed_opt_us,error_constVel,elapsed_consVel_us,optim_iterations\n";

# Recorre todos los archivos CSV en la carpeta
for filename in sorted(os.listdir(results_dir)):
    if filename.endswith(".csv"):
        path = os.path.join(results_dir, filename)
        seq_name = filename.replace("results_", "").replace(".csv", "")
        
        # Leer CSV y limpiar nombres de columnas
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        
        # Calcular promedios
        mean_small = df["error_small"].mean()
        mean_wo_preint = df["error_wo_preint"].mean()
        mean_opt = df["error_opt"].mean()
        mean_const = df["error_constVel"].mean()
        mean_elapsed_small_us = df["elapsed_small_us"].mean()
        mean_elapsed_wo_preint_us = df["elapsed_wo_preint_us"].mean()

        mean_elapsed_opt_us = df["elapsed_opt_us"].mean()
        mean_elapsed_consVel_us = df["elapsed_consVel_us"].mean()
        mean_optim_iterations = df["optim_iterations"].mean()

        # Elegir la mejor (mínima)
        best_method = min({
            "Closed-form": mean_small,
            "Nonlin optim.": mean_opt,
            "Const. vel.": mean_const,
            "w/o preint.": mean_wo_preint
        }, key=lambda k: {
            "Closed-form": mean_small,
            "Nonlin optim.": mean_opt,
            "Const. vel.": mean_const,
            "w/o preint.": mean_wo_preint
        }[k])

        results.append({
            "Seq. Name": seq_name,
            "Closed-form": mean_small,
            "Nonlin optim.": mean_opt,
            "Const. vel.": mean_const,
            "w/o preint.": mean_wo_preint,
            "Bold Index": best_method,
            "Cost Closed-form": mean_elapsed_small_us,
            "Cost Optim": mean_elapsed_opt_us,
            "Cost Vel. const.": mean_elapsed_consVel_us,
            "Cost w/o preint.": mean_elapsed_wo_preint_us,
            "Optim. iter": mean_optim_iterations,
        })

# Calcular promedios globales
mean_row = {
    "Seq. Name": "Mean value",
    "Closed-form": np.mean([r["Closed-form"] for r in results]),
    "Nonlin optim.": np.mean([r["Nonlin optim."] for r in results]),
    "Const. vel.": np.mean([r["Const. vel."] for r in results]),
    "w/o preint.": np.mean([r["w/o preint."] for r in results]),
    "Bold Index": None,
    "Cost Closed-form": np.mean([r["Cost Closed-form"] for r in results]),
    "Cost Optim": np.mean([r["Cost Optim"] for r in results]),
    "Cost Vel. const.": np.mean([r["Cost Vel. const."] for r in results]),
    "Cost w/o preint.": np.mean([r["Cost w/o preint."] for r in results]),
    "Optim. iter": np.mean([r["Optim. iter"] for r in results])
}
results.append(mean_row)

# Crear DataFrame final
df_results = pd.DataFrame(results)

# Mostrar como tabla de texto
print(df_results.to_string(index=False, float_format="%.6f"))
