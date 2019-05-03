#!/usr/bin/env ruby

require 'csv'

result_dir = ARGV[0]
best_allcases = ARGV[1]
result_file = "pmlog.csv"
files = Array.new
results = Hash.new { |hash, key| hash[key] = {} }
files = Dir.glob("./#{result_dir}/*.log")
puts files

files.each do |f|
  puts "---------------------------"
  puts f

  peak_tflops = 0
  cmd = "cat #{f}  | grep Fastest | awk '{print $2}'"
  peak_gflops = `#{cmd}`.to_f
  cmd = "cat #{f}  | grep Fastest | awk '{print $8}'"
  fastkernel_name = `#{cmd}`
  cmd = "cat #{f}  | grep -Ei 'Problem.*: [0-9]+,' | awk '{print $2,$3,$5}'"
  problem_size = `#{cmd}`
  #cmd = "cat #{f}  | grep PASSED | awk '{print $3 }'"
  #kernel_name = `#{cmd}`
  
  pm_file = f.sub(/\.log/,'') +"_pmlog.csv"

# power vddcr_gfx: Z (26)
# power vddcr_soc: AH (26 + 8 = 34)
# power vddio_mem: AO (26 + 15 = 41)
# power vddci_mem: AV (26 + 22 = 48)

  #tmp = `cat #{pm_file} | cut -d ',' -f 3,6,7,20,109,110`.split("\n")
  tmp = `cat #{pm_file} | cut -d ',' -f 3,6,7,20,26,34,41,48,109,110`.split("\n")
  
  tmp1 = tmp.sort_by { |l| l.split(",")[0].to_f}.reverse
  tmp2 = tmp1[0].split(",")

  peak_power = tmp2[0].to_f
  temperature = tmp2[1].to_f
  hbm_tmp = tmp2[2].to_f
  vddgfx = tmp2[3].to_f
  power_vddcr_gfx = tmp2[4].to_f
  power_vddcr_soc = tmp2[5].to_f
  power_vddio_mem = tmp2[6].to_f
  power_vddci_mem = tmp2[7].to_f

  fset = tmp2[8].to_f
  fload = tmp2[9].to_f

  measured_gflops = peak_gflops.round(2)
  software_eff = (100*peak_gflops.to_f/(4096*fload/1000)).round(2)
  silicon_eff = (100*fload/fset).round(2)
  overall_eff = (100*peak_gflops.to_f/(4096*fset/1000)).round(2)

  puts "kernel_name: #{fastkernel_name}" 
  puts "problem_size: #{problem_size}" 
  puts "peak power: #{peak_power}"
  puts "vddgfx: #{vddgfx}"
  puts "temperature: #{temperature}"
  puts "hbm temperature: #{hbm_tmp}"
  puts "fload: #{fload }"
  puts "gflops fload: #{4096*fload/1000}"
  puts "gflops fset: #{4096*fset/1000}"
  puts "silicon eff: #{silicon_eff}"
  puts "overall eff: #{overall_eff}"
  puts "software eff: #{software_eff}"

  fastkernel_name.delete!("\n")
  problem_size.delete!("\n")
  problem_size.delete!(",")
  results[f]["kernel_name"]=fastkernel_name
  results[f]["problem size"]= problem_size
  results[f]["fset"] = fset
  results[f]["peak_power"] = peak_power
  results[f]["power_vddcr_gfx"] = power_vddcr_gfx
  results[f]["power_vddcr_soc"] = power_vddcr_soc
  results[f]["power_vddio_mem"] = power_vddio_mem
  results[f]["power_vddci_mem"] = power_vddci_mem
  results[f]["temperature"] = temperature
  results[f]["fload"] = fload
  results[f]["measured_gflops"] = measured_gflops
  results[f]["software_eff"] = software_eff
  results[f]["silicon_eff"] = silicon_eff
  results[f]["overall_eff"] = overall_eff
  #results[f]["hbm_tmp"] = hbm_tmp 
  
end



cols = ["KernelName", "ProblemSize", "Fset (MHz)","Peak Power (W)", "Power VDDCR GFX", "Power VDDCR SOC", "Power VDDIO MEM", "Power VDDCI MEM",	"Temperature C", "Fload (MHz)",	"Measured GFlops",	"Software Efficiency (%)",	"Silicon Efficiency (%)",	"Overall Efficiency (%)"]


CSV.open("#{result_file}", "w") do |csv|
  csv << cols
  results.each do |k,v|
  
    row_tmp = Array.new
#    row_tmp << k

    v.each do |kk, vv|
      row_tmp << vv
    end
    csv << row_tmp
  end

end
