# ----------------------------------在测试集上评估模型的性能, 计算准确率和平均损失----------------------------------
import torch
from torch.utils.data import DataLoader


class test_accuracy:

    def test_accuracy(self, net, parameters, testDataset, dev, lossFun,config):
        # 存储损失

        testDataLoader=DataLoader(testDataset, batch_size=config["batchSize"], shuffle=True)
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        net.load_state_dict(parameters, strict=True)
        net.eval()  # 确保模型处于评估模式
        with torch.no_grad():
            for data, label in testDataLoader:
                data, label = data.to(dev), label.to(dev)
                output_train,output_test = net(data)#output_train未经过softmax，out_test经过了softmax
                # 计算损失
                loss = lossFun(output_test, label)
                batch_loss = loss.item()
                batch_size = label.size(0)
                total_loss += batch_loss * batch_size  # 累加加权损失
                # 计算正确数
                preds = torch.argmax(output_test, dim=1)
                correct = (preds == label).sum().item()
                total_correct += correct
                total_samples += batch_size
        correctRate = total_correct / total_samples
        lossAvg = total_loss / total_samples
        return lossAvg,correctRate


        # loss_collector = []
        # with torch.no_grad():
        #     net.load_state_dict(parameters, strict=True)
        #     sum_accu = 0
        #     num = 0
        #     loss_collector.clear()
        #     # 载入测试集
        #     for data, label in testDataLoader:
        #         data, label = data.to(dev), label.to(dev)
        #         output = net(data)
        #         print(output)
        #         loss = lossFun(output, label)
        #         # loss = 1
        #         loss_collector.append(loss.item())
        #         output = torch.argmax(output, dim=1)
        #         sum_accu += (output == label).float().mean()
        #         print(sum_accu)
        #         num += 1
        #
        #     accuracy = sum_accu / num
        #     avg_loss = sum(loss_collector) / len(loss_collector)
        # return avg_loss, accuracy