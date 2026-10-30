module github.com/openfluke/welvet_sweep_2_snat

go 1.27.0

require (
	github.com/openfluke/snat v1.1.0
	github.com/openfluke/welvet v1.1.3
)

require github.com/openfluke/webgpu v1.0.5 // indirect

replace github.com/openfluke/snat => ../snat
