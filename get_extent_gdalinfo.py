
import subprocess

reference_tif = r"E:\Processing\InSAR_multivariate_analysis\variables_data\dsw-1984-2023-frequency_masked.tif"

result = subprocess.run(["gdalinfo", reference_tif], capture_output=True, text=True)
print(result.stdout)
