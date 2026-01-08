from pymodbus.client.sync import ModbusSerialClient as ModbusClient
import time

# 串口配置
# serial_port = '/dev/ttyCH9344USB1'  # 串口号
# baudrate = 115200     # 波特率
# parity = 'N'          # 无校验
# stopbits = 1          # 停止位
# bytesize = 8          # 数据位


class ZXGripper:
    def __init__(self, 
                 serial_port: str = "/dev/ttyCH9344USB1",
                 baudrate: int = 115200,
                 parity: str = "N",
                 stopbits: int = 1,
                 byte_size: int = 8) -> None:
        
        self.client = ModbusClient(
                    method='rtu',
                    port=serial_port,
                    baudrate=baudrate,
                    parity=parity,
                    stopbits=stopbits,
                    bytesize=byte_size
                )
        self.start()
        

    def send_register_value(self, register_address, value):
        """向指定寄存器发送指定值"""
        if self.client.connect():
            print(f"连接成功，准备向寄存器 {hex(register_address)} 发送值 {value}...")

            # 确保值在 16 位范围内（-32768 到 32767）
            if value < -32768 or value > 32767:
                print(f"值 {value} 超出有效范围！")
                self.client.close()
                return

            # 向指定寄存器写入指定值
            result = self.client.write_register(register_address, value, unit=1)

            if result.isError():
                print(f"写入寄存器 {hex(register_address)} 数据失败:", result)
            else:
                print(f"寄存器 {hex(register_address)} 数据已写入，值: {value}")
        else:
            print("无法连接到 Modbus 从站")

        # 关闭 Modbus 连接
        self.client.close()


    def start(self):
        # 调用函数使能夹爪
        self.send_register_value(0X0100, 1)
        # 张开速度
        self.send_register_value(0X0102, 100)
        # 张开力矩
        self.send_register_value(0X0103, 100)
        # 张开给位置
        self.send_register_value(0X0101, 1000)

        # 闭合给位置
        self.send_register_value(0X0105, 1000)
        # 闭合速度
        self.send_register_value(0X0106, 100)
        # 闭合力矩
        self.send_register_value(0X0107, 100)

    def open_gripper(self):
        self.send_register_value(0X0104, 1)

    def close_gripper(self):
        self.send_register_value(0X0108, 1)

    def set_open_width(self, width):
        self.send_register_value(0X0101, width)
        self.open_gripper()

    def set_close_width(self, width):
        self.send_register_value(0X0105, width)
        self.open_gripper()



if __name__ == "__main__":
    zx = ZXGripper()
    zx.start()
    zx.close_gripper()
    zx.open_gripper()