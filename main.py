from data_manager import DataManager
from utils.toolkits import seed_set, load_json
import utils.toolkits as toolkits

seed_set()

def main():
    # 配置文件选择开关
    # CONFIG_TYPE = "best_cifar_in21k_ncmloss"
    # CONFIG_TYPE = "cifar_in21k_ncmloss"

    # CONFIG_TYPE = "ina_in21k_ncmloss"
    # CONFIG_TYPE = "best_ina_in21k_ncmloss"

    CONFIG_TYPE = "vtab_in21k_ncmloss"
    # CONFIG_TYPE = "best_vtab_in21k_ncmloss"

    # CONFIG_TYPE = "inr_in21k_ncmloss"
    # CONFIG_TYPE = "best_inr_in21k_ncmloss"

    config_map = {
        "cifar_in21k_ncmloss": "exps/cifar/cifar_in21k_ncmloss.json",
        "best_cifar_in21k_ncmloss": "exps/cifar/best/cifar_in21k_ncmloss.json",

        "ina_in21k_ncmloss": "exps/ina/ina_in21k_ncmloss.json",
        "best_ina_in21k_ncmloss": "exps/ina/best/best_ina_in21k_ncmloss.json",

        "vtab_in21k_ncmloss": "exps/vtab/vtab_in21k_ncmloss.json",
        "best_vtab_in21k_ncmloss": "exps/vtab/best/best_vtab_in21k_ncmloss.json",

        "inr_in21k_ncmloss": "exps/inr/inr_in21k_ncmloss.json",
        "best_inr_in21k_ncmloss": "exps/inr/best/best_inr_in21k_ncmloss.json",
    }

    # 指定配置文件路径
    config_path = config_map[CONFIG_TYPE]

    # 加载配置参数
    args = load_json(config_path)  # 直接获取字典格式参数

    # 初始化数据管理器
    if True:
        data_manager = DataManager(
            dataset_name=args["dataset"],
            shuffle=True,
            seed=args["seed"],
            init_cls=args["init_cls"],
            increment=args["increment"],
            args=args
        )

    # 初始化模型
    model = toolkits.get_model(model_name=args["model_name"], args=args)

    # 执行增量学习任务
    for task in range(len(data_manager._increments)):
        print(f'Here Comes Task{task}', '*'*50)
        model.incremental_train(data_manager)
        model.eval_accuracy(words=f'{task}')
        # model.watch_cosine_similarity()
        model.after_task()

if __name__ == '__main__':
    main()