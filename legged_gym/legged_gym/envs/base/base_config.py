# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import inspect  # 导入 inspect 标准库模块，用于判断某个属性是否是一个“类对象”。

class BaseConfig:
    def __init__(self) -> None:  # BaseConfig 实例化时会自动调用该构造函数，用来初始化配置对象内部的嵌套配置类。
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""  # 说明该函数会递归初始化成员类，目的是把嵌套 class 变成可访问的配置实例。
        self.init_member_classes(self)  # 从当前配置对象 self 开始，递归扫描并实例化它内部定义的所有嵌套 class。
    
    @staticmethod
    def init_member_classes(obj):  # 定义静态方法，用于递归处理任意配置对象中的嵌套 class，不依赖 BaseConfig 实例状态。
        # iterate over all attributes names
        for key in dir(obj):  # 遍历 obj 上所有可访问属性名，包括用户定义的配置项和 Python 内置属性。
            # disregard builtin attributes
            # if key.startswith("__"):
            if key=="__class__":  # 跳过 __class__ 属性，避免把对象自身的类型当成普通配置成员处理。
                continue  # 当前属性不参与处理，直接进入下一个属性。
            # get the corresponding attribute object
            var =  getattr(obj, key)  # 根据属性名取出真实属性对象，例如 env、terrain、control 这些嵌套 class。
            # check if it the attribute is a class
            if inspect.isclass(var):  # 如果该属性本身是一个类，说明它是配置文件中用 class 写出的嵌套配置块。
                # instantate the class
                i_var = var()  # 实例化这个嵌套配置类，把 class env 变成 env 对象。
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)  # 用实例化后的对象替换原来的类对象，这样后续可以通过 cfg.env.xxx 访问和修改参数。
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)  # 继续递归处理该嵌套配置对象内部可能存在的更深层嵌套 class。
