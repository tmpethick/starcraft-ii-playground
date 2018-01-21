init:
	conda env create -f environment.yml

update:
	conda env update -f environment.yml

generate:
	python -m pysc2.bin.agent --map CollectMineralShards --agent scripted.CollectMineralShards

.PHONY: generate
