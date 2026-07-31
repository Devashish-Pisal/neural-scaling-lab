METRICS = {

    # "metric-ViT-Ti-16_imgnet_fg-0-100_ep-300_steps-186k"
    "vit-ti/16":{
        "eval/number of parameters":5721832,
        "eval/macs":1076704704,
        "eval/flops":1461060180,
        "eval/inference_memory_@128":505310720,
        "eval/inference_memory_@64":427847168,
        "eval/inference_memory_@32":389115392,
        "eval/inference_memory_@16":369749504,
        "eval/inference_memory_@1":352712192,
        "eval/training_time":{
            "batch_size":512,
            "step_time_ms":1062.9527777777778
        },
        "eval/peak_memory":10529998848,
        "eval/throughput":{
            "batch_size":4096,
            "value":16927.09742851711
        }
    },

    # "metric-ViT-S-16_imgnet_fg-0-100_ep-300_steps-186k"
    "vit-s/16":{
        "eval/number of parameters":22059496,
        "eval/macs":4244925312,
        "eval/flops":5013636264,
        "eval/inference_memory_@128":759194112,
        "eval/inference_memory_@64":604103168,
        "eval/inference_memory_@32":526688768,
        "eval/inference_memory_@16":487899648,
        "eval/inference_memory_@1":452815360,
        "eval/training_time":{
            "batch_size":512,
            "step_time_ms":1068.299045138889
        },
        "eval/peak_memory":20355668992,
        "eval/throughput":{
            "batch_size":4096,
            "value":8639.307676001978
        }
    },

    # "metric-ViT-S-32_imgnet_fg-0-100_ep-300_steps-186k"
    "vit-s/32":{
        "eval/number of parameters":22887784,
        "eval/macs":1120810752,
        "eval/flops":1177793568,
        "eval/inference_memory_@128":567934464,
        "eval/inference_memory_@64":494268416,
        "eval/inference_memory_@32":474181632,
        "eval/inference_memory_@16":464737280,
        "eval/inference_memory_@1":457698304,
        "eval/training_time":{
            "batch_size":512,
            "step_time_ms":898.8497395833333
        },
        "eval/peak_memory":6084318720,
        "eval/throughput":{
            "batch_size":16384,
            "value":27230.346941781587
        }
    },

    # "metric-ViT-B-16_imgnet_fg-0-100_ep-300_steps-186k"
    "vit-b/16":{
        "eval/number of parameters":86585320,
        "eval/macs":16855914240,
        "eval/flops":18393336144,
        "eval/inference_memory_@128":1464093184,
        "eval/inference_memory_@64":1149159936,
        "eval/inference_memory_@32":992692736,
        "eval/inference_memory_@16":914016768,
        "eval/inference_memory_@1":854908416,
        "eval/training_time":{
            "batch_size":512,
            "step_time_ms":908.9777777777778
        },
        "eval/peak_memory":40423478272,
        "eval/throughput":{
            "batch_size":1024,
            "value":3521.77556177109
        }
    },

    # "metric-ViT-B-28_imgnet_fg-0-100_ep-300_steps-186k"
    "vit-b/28": {
        "eval/number of parameters": 87700456,
        "eval/macs": 5639572224,
        "eval/flops": 5824372176,
        "eval/inference_memory_@128": 1051474432,
        "eval/inference_memory_@64": 949107200,
        "eval/inference_memory_@32": 895564288,
        "eval/inference_memory_@16": 874332160,
        "eval/inference_memory_@1": 857432576,
        "eval/training_time": {
            "batch_size": 512,
            "step_time_ms": 893.1580729166667
        },
        "eval/peak_memory": 14791792640,
        "eval/throughput": {
            "batch_size": 8192,
            "value": 10049.82455120593
        }
    },
}