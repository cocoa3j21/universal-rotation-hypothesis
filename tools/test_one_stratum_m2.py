import sys, time
sys.path.insert(0, 'Labs/universal-rotation-hypothesis/tools')
import run_model2_h0_calibration as m2cal

strata = m2cal.generate_strata()
t0 = time.time()
res = m2cal.process_stratum_h0(strata[0])
t1 = time.time()
print(f"One stratum took {t1-t0:.2f} s. snr_th={res['_snr_threshold']:.2f}, fpr={res['fpr_h0']:.4f}")
