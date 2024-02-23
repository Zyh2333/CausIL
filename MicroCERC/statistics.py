import os


def print_pr(nums):
    pr1 = 0
    pr2 = 0
    pr3 = 0
    pr4 = 0
    pr5 = 0
    pr6 = 0
    pr7 = 0
    pr8 = 0
    pr9 = 0
    pr10 = 0
    fill_nums = []
    for num in nums:
        # if num != 0 and num < 10:
        if num != 0:
            fill_nums.append(num)
    for num in fill_nums:
        if num <= 10:
            pr10 += 1
            if num <= 9:
                pr9 += 1
                if num <= 8:
                    pr8 += 1
                    if num <= 7:
                        pr7 += 1
                        if num <= 6:
                            pr6 += 1
                            if num <= 5:
                                pr5 += 1
                                if num <= 4:
                                    pr4 += 1
                                    if num <= 3:
                                        pr3 += 1
                                        if num <= 2:
                                            pr2 += 1
                                            if num == 1:
                                                pr1 += 1
    pr_1 = round(pr1 / len(fill_nums), 3)
    pr_2 = round(pr2 / len(fill_nums), 3)
    pr_3 = round(pr3 / len(fill_nums), 3)
    pr_4 = round(pr4 / len(fill_nums), 3)
    pr_5 = round(pr5 / len(fill_nums), 3)
    pr_6 = round(pr6 / len(fill_nums), 3)
    pr_7 = round(pr7 / len(fill_nums), 3)
    pr_8 = round(pr8 / len(fill_nums), 3)
    pr_9 = round(pr9 / len(fill_nums), 3)
    pr_10 = round(pr10 / len(fill_nums), 3)
    print('PR@1:' + str(pr_1))
    print('PR@2:' + str(pr_2))
    print('PR@3:' + str(pr_3))
    print('PR@5:' + str(pr_5))
    print('PR@10:' + str(pr_10))
    avg_1 = pr_1
    avg_2 = round((pr_1 + pr_2) / 2, 3)
    avg_3 = round((pr_1 + pr_2 + pr_3) / 3, 3)
    avg_5 = round((pr_1 + pr_2 + pr_3 + pr_4 + pr_5) / 5, 3)
    avg_10 = round((pr_1 + pr_2 + pr_3 + pr_4 + pr_5 + pr_6 + pr_7 + pr_8 + pr_9 + pr_10) / 10, 3)
    print('AVG@1:' + str(avg_1))
    print('AVG@2:' + str(avg_2))
    print('AVG@3:' + str(avg_3))
    print('AVG@5:' + str(avg_5))
    print('AVG@10:' + str(avg_10))
    return pr_1, pr_3, pr_5, pr_10, avg_1, avg_3, avg_5, avg_10


def read_logs(directory, top_k_list, epoch_list):
    if top_k_list is None:
        top_k_list = []
    if epoch_list is None:
        epoch_list = []
    # 遍历指定文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 只处理以.log结尾的文件
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        # 如果文件为空则跳过
                        if not lines:
                            continue
                        # 获取最后一行并判断是否满足条件
                        last_line = lines[-1]
                        if last_line.startswith('top_k:'):
                            _, number_str = last_line.split(':')
                            number = int(number_str.strip())
                            top_k_list.append(number)
                        for line in lines:
                            if line.startswith('Early stopping with epoch'):
                                epoch = int(line.split(',')[0].split(':')[1].strip())
                                epoch_list.append(epoch)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")


def read_effi_logs(directory, top_k_list, time_list, loss):
    if top_k_list is None:
        top_k_list = []
    if time_list is None:
        time_list = []
    # 遍历指定文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 只处理以.log结尾的文件
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        # 如果文件为空则跳过
                        if not lines:
                            continue
                        is_loss_line = False
                        for line in lines:
                            if line.startswith(loss):
                                time = int(line.split(':')[-1].strip())
                                time_list.append(time)
                                is_loss_line = True
                            else:
                                if is_loss_line:
                                    if line.startswith('top_k:'):
                                        _, number_str = line.split(':')
                                        number = int(number_str.strip())
                                        top_k_list.append(number)
                                        break
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")


def read_specific_logs(directory, top_k_list):
    if top_k_list is None:
        top_k_list = []
    # 遍历指定文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 只处理以.log结尾的文件
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        # 如果文件为空则跳过
                        if not lines:
                            continue
                        is_begin = False
                        root_cause = None
                        count = 0
                        for line in lines:
                            if 'root_cause' in line:
                                root_cause = line.split(":")[1].strip()
                                continue
                            if not is_begin and line.startswith('top_k:'):
                                is_begin = True
                            elif is_begin:
                                node = line.split(":")[0]
                                if 'details-v1' in node or 'ratings-v1' in node or 'productpage-v1' in node or 'reviews-v1' in node or 'reviews-v2' in node or 'reviews-v3' in node:
                                    count += 1
                                if ('edge' in root_cause and root_cause in node) or (
                                        'edge' not in root_cause and root_cause in node and 'edge' not in node):
                                    top_k_list.append(count)
                                    break
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")


# 测试代码
top_k_list = []
epoch_list = []
time_list = []
directory = '/Users/zhuyuhan/Documents/159-WHU/论文投递/MicroCERC/result/result-0.0x/result-0.07'
# directory = '/Users/zhuyuhan/Documents/159-WHU/论文投递/MicroCERC/result/result-lstm'
# directory = '/Users/zhuyuhan/Documents/159-WHU/论文投递/MicroCERC/result/result-losses/result-51e-6-0.07'
# directory = '/Users/zhuyuhan/Documents/159-WHU/论文投递/MicroCERC/result/result-ablation-self'
# directory = '/Users/zhuyuhan/Documents/159-WHU/论文投递/MicroCERC/result/result-spaces-2'
# directory = '/Users/zhuyuhan/Documents/159-WHU/论文投递/MicroCERC/result/result20240220'
# directory = '/Users/zhuyuhan/Documents/159-WHU/论文投递/MicroCERC/PC'
# directory = '/Users/zhuyuhan/Documents/159-WHU/论文投递/MicroCERC/microRCA_result'
# read_logs(directory, top_k_list, epoch_list)
read_specific_logs(directory, top_k_list)
# read_effi_logs(directory, top_k_list, time_list, '1e-6')
# print("Numbers extracted from log files:", top_k_list)
print_pr(top_k_list)
# print(str(sum(time_list) / len(time_list)) + 's')
# print(str(sum(epoch_list) / len(epoch_list) * 2) + 's')

# nums = [2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 2, 4, 4, 2, 2, 4, 4, 4, 1]
# CausalRCA C
# nums = [2, 5, 2, 11, 4, 2, 11, 11, 4, 11, 3, 2, 11, 4, 2, 3, 1, 3, 3, 3]
# print_pr(nums)

# CausalRCA D
# nums = [5, 11, 2, 11, 3, 2, 5, 3, 2, 4, 3, 2, 3, 11, 5, 1, 11, 3, 2, 4]
# print_pr(nums)

# MicroCERC gru C
# nums = [2, 2, 2, 1, 3, 2, 2, 1, 2, 2, 3, 2, 4, 4, 2, 1, 11, 3, 4, 1]
# print_pr(nums)

# MicroCERC lstm bookinfo
# nums = [1, 4, 2, 1, 8, 3, 8, 4, 6, 4, 5]
# print_pr(nums)
