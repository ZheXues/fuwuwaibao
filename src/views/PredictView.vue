<template>
    <el-element>
        <app-header></app-header>
        <div class="main">
            <div class="main-header">
                <p class="tital">LOGO医疗保险欺诈预测系统</p>
                <router-link class="el-icon-close" to="/"></router-link>
            </div>
            <div class="container">
                <table id="predictTable" :data="predictData">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>预测结果</th>
                            <th>正向概率</th>
                            <th>负向概率</th>
                        </tr>
                    </thead>
                    <tbody id="predictList">
                    </tbody>
                </table>
                <div class="button-box">
                    <button class="button" id="button3" @click="downloadFiles">下载结果</button>
                    <router-link to="/predict/details" class="button" id="button4">详细信息</router-link>
                </div>
            </div>
        </div>
    </el-element>
</template>

<script>
// import axios from "axios";
//这里导入其他文件
export default {
    //引入的组件注入到对象中才能使用
    data() {
        return {
            predictList: [],
        };
    },
    methods: {
        // 下载结果
        downloadFiles() { }
    }, //方法集合
    mounted() { // 显示数据

    }, //生命周期 - 挂载完成
};
</script>

<style scoped></style>