.mode csv
.output SM-Active.csv
SELECT timestamp, value FROM gpu_metrics WHERE metricid = (SELECT metricid FROM TARGET_INFO_GPU_METRICS WHERE metricName = 'SM Active');
