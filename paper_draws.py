import matplotlib.pyplot as plt

# 重新创建子图布局，并保留横纵坐标标签（label）


def create_subplots_with_labels():
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 子图1: Output 1 - Current Id
    axs[0, 0].set_title('Output 1: Current Id')
    axs[0, 0].set_xlabel('Sample Points')
    axs[0, 0].set_ylabel('Current Id')
    # 设置图例的占位标签
    axs[0, 0].plot([], [], 'b', label='Actual Current Id')
    axs[0, 0].plot([], [], 'r', label='Predict Current Id')
    axs[0, 0].legend()

    # 子图2: Output 2 - Current Iq
    axs[0, 1].set_title('Output 2: Current Iq')
    axs[0, 1].set_xlabel('Sample Points')
    axs[0, 1].set_ylabel('Current Iq')
    # 设置图例的占位标签
    axs[0, 1].plot([], [], 'b', label='Actual Current Iq')
    axs[0, 1].plot([], [], 'r', label='Predict Current Iq')
    axs[0, 1].legend()

    # 子图3: Output 3 - Speed
    axs[1, 0].set_title('Output 3: Speed')
    axs[1, 0].set_xlabel('Sample Points')
    axs[1, 0].set_ylabel('Speed')
    # 设置图例的占位标签
    axs[1, 0].plot([], [], 'b', label='Actual Speed')
    axs[1, 0].plot([], [], 'r', label='Predict Speed')
    axs[1, 0].legend()

    # 子图4: Output 4 - Angle
    axs[1, 1].set_title('Output 4: Angle')
    axs[1, 1].set_xlabel('Sample Points')
    axs[1, 1].set_ylabel('Angle')
    # 设置图例的占位标签
    axs[1, 1].plot([], [], 'b', label='Actual Angle')
    axs[1, 1].plot([], [], 'r', label='Predict Angle')
    axs[1, 1].legend()

    # 调整子图间距
    plt.tight_layout()
    return fig, axs


if __name__ == '__main__':

    # 创建带标签的子图布局
    fig, axs = create_subplots_with_labels()

    # 保存图表为文件
    output_file_path_with_labels = './pic/paper.png'
    plt.savefig(output_file_path_with_labels)

    # 显示图表
    plt.show()
