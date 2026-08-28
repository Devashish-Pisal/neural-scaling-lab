METRICS = {

    # "metric-ViT-Ti-16_imgnet_fg-0-100_ep-300_steps-186k"
    "vit-ti/16":{
        "eval/number of parameters":5721832, # static model property, always same
        "eval/macs":1076704704, # macs for 1 image
        "eval/flops":1461060180, # flops for 1 image
        "eval/inference_memory_@128":505310720, # inference memory for 128 images
        "eval/inference_memory_@64":427847168, # inference memory for 64 images
        "eval/inference_memory_@32":389115392, # inference memory for 32 images
        "eval/inference_memory_@16":369749504, # inference memory for 16 images
        "eval/inference_memory_@1":352712192, # inference memory for 1 image
        "eval/training_time":{
            "batch_size":512,
            "step_time_ms":1062.9527777777778 # step time in milliseconds for 512 images
        },
        "eval/peak_memory":10529998848, # peak memory usage for 512 images for 1 GPU (used global batch size is 2048 with 4 GPUs)
        "eval/throughput":{
            "batch_size":4096,
            "value":16927.09742851711 # images/sec for batch size 4096
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

    # "metric-ViT-B-32_imgnet_fg-0-100_ep-300_steps-186k"
    "vit-b/32" : {
        "eval/number of parameters": 88241896,
        "eval/macs": 4364987904,
        "eval/flops": 4478953536,
        "eval/inference_memory_@128": 1004897792,
        "eval/inference_memory_@64": 923764224,
        "eval/inference_memory_@32": 883656192,
        "eval/inference_memory_@16": 873024512,
        "eval/inference_memory_@1": 861141504,
        "eval/training_time": {
            "batch_size": 512,
            "step_time_ms": 861.1502604166667
        },
        "eval/peak_memory": 11897003008,
        "eval/throughput": {
            "batch_size": 2048,
            "value": 11823.750345137212,
        }
    },

    # "metric-ViT-L-16_imgnet_fg-0-100_ep-300_steps-186k_bs-256"
    # NOTE: BATCH SIZE IS REDUCED, BECAUSE L MODEL VARIANT WAS NOT FITTING ON A100 PARTITION
    "vit-l/16": {
      "eval/number of parameters": 304374760,
      "eval/macs": 59666740224,
      "eval/flops": 63765524352,
      "eval/inference_memory_@128": 2809778688,
      "eval/inference_memory_@64": 2394542592,
      "eval/inference_memory_@32": 2187547136,
      "eval/inference_memory_@16": 2083869184,
      "eval/inference_memory_@1": 1995612672,
      "eval/training_time": {
        "batch_size": 256,
        "step_time_ms": 823.8171006944444
      },
      "eval/peak_memory": 55178507776,
      "eval/throughput": {
        "batch_size": 1024,
        "value": 1169.7198941988966
      }
    },

    # "metric-ViT-L-32_imgnet_fg-0-100_ep-300_steps-186k"
    # NOTE: BATCH SIZE IS REDUCED, BECAUSE L MODEL VARIANT WAS NOT FITTING ON A100 PARTITION
    "vit-l/32": {
        "eval/number of parameters": 306583528,
        "eval/macs": 15259625472,
        "eval/flops": 15563278848,
        "eval/inference_memory_@128": 2203132416,
        "eval/inference_memory_@64": 2100109824,
        "eval/inference_memory_@32": 2054234624,
        "eval/inference_memory_@16": 2028022272,
        "eval/inference_memory_@1": 2004165120,
        "eval/training_time": {
            "batch_size": 256,
            "step_time_ms": 579.2594618055556
        },
        "eval/peak_memory": 17657423360,
        "eval/throughput": {
            "batch_size": 16384,
            "value": 4373.902265893363
        }
    }
}