<template>
  <a-modal
    :open="open"
    title="添加附件"
    ok-text="添加附件"
    cancel-text="取消"
    :confirm-loading="confirming"
    :ok-button-props="{ disabled: confirmDisabled }"
    @ok="handleConfirm"
    @cancel="handleCancel"
  >
    <a-upload-dragger
      :multiple="true"
      :show-upload-list="false"
      :before-upload="handleBeforeUpload"
      :disabled="confirming"
      class="attachment-dropzone"
    >
      <p class="dropzone-title">点击或拖拽文件到此处上传</p>
      <p class="dropzone-desc">支持任意文件格式 ≤ 5 MB；PDF 和图片可选解析为 Markdown。</p>
    </a-upload-dragger>

    <div v-if="fileItems.length" class="attachment-list">
      <div v-for="item in fileItems" :key="item.localId" class="attachment-item">
        <div class="attachment-item-main">
          <div class="attachment-name-row">
            <span class="attachment-name">{{ item.fileName }}</span>
            <a-tag :color="getStatusColor(item.status)">{{ getStatusLabel(item.status) }}</a-tag>
          </div>
          <div class="attachment-meta">
            <span>{{ formatFileSize(item.fileSize) }}</span>
            <span v-if="item.error" class="attachment-error">{{ item.error }}</span>
            <span v-else-if="item.parseError" class="attachment-error">{{ item.parseError }}</span>
            <span v-else-if="item.parsedObjectName" class="attachment-parsed">已生成解析附件</span>
          </div>
        </div>

        <div class="attachment-actions">
          <template v-if="item.parseSupported && item.status !== 'uploading' && item.status !== 'error'">
            <a-select
              v-model:value="item.selectedParseMethod"
              :options="getParseMethodOptions(item)"
              size="small"
              class="parse-select"
              :disabled="item.status === 'parsing' || confirming"
              @change="(value) => handleParseMethodChange(item.localId, value)"
            />
            <a-button
              size="small"
              :loading="item.status === 'parsing'"
              :disabled="!item.selectedParseMethod || confirming"
              @click="handleParse(item)"
            >
              解析
            </a-button>
          </template>
          <a-button size="small" type="text" :disabled="confirming" @click="removeItem(item.localId)">
            移除
          </a-button>
        </div>
      </div>
    </div>
  </a-modal>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { message } from 'ant-design-vue'
import { threadApi } from '@/apis'

const props = defineProps({
  open: { type: Boolean, default: false },
  threadId: { type: String, default: '' },
  ensureThread: { type: Function, default: null }
})

const emit = defineEmits(['update:open', 'added'])

const fileItems = ref([])
const confirming = ref(false)
let localIdSeed = 0

const methodLabels = {
  disable: 'PDF 文本提取',
  rapid_ocr: 'RapidOCR',
  mineru_ocr: 'MinerU OCR',
  mineru_official: 'MinerU Official',
  pp_structure_v3_ocr: 'PP-Structure V3',
  deepseek_ocr: 'DeepSeek OCR'
}

const busy = computed(() => fileItems.value.some((item) => ['uploading', 'parsing'].includes(item.status)))
const confirmableItems = computed(() =>
  fileItems.value.filter((item) => ['uploaded', 'parsed'].includes(item.status))
)
const confirmDisabled = computed(() => busy.value || confirmableItems.value.length === 0)

watch(
  () => props.open,
  (open) => {
    if (!open) {
      fileItems.value = []
      confirming.value = false
    }
  }
)

const getErrorMessage = (error, fallback = '操作失败') => {
  return error?.response?.data?.detail || error?.message || fallback
}

const getDefaultParseMethod = (methods) => {
  if (!Array.isArray(methods) || methods.length === 0) return null
  return methods.includes('disable') ? 'disable' : methods[0]
}

const normalizeTmpUpload = (response) => ({
  tmpFileId: response.tmp_file_id,
  fileName: response.file_name,
  fileType: response.file_type,
  fileSize: response.file_size,
  bucketName: response.bucket_name,
  objectName: response.object_name,
  minioUrl: response.minio_url,
  parseSupported: response.parse_supported,
  parseMethods: response.parse_methods || [],
  selectedParseMethod: getDefaultParseMethod(response.parse_methods || [])
})

const updateItem = (localId, patch) => {
  fileItems.value = fileItems.value.map((item) =>
    item.localId === localId ? { ...item, ...patch } : item
  )
}

const uploadFile = async (file) => {
  const localId = `${Date.now()}-${localIdSeed++}`
  const item = {
    localId,
    fileName: file.name,
    fileSize: file.size,
    status: 'uploading',
    error: null,
    parseError: null,
    parseSupported: false,
    parseMethods: []
  }
  fileItems.value.push(item)

  try {
    const response = await threadApi.uploadTmpAttachment(file)
    updateItem(localId, { ...normalizeTmpUpload(response), status: 'uploaded' })
  } catch (error) {
    updateItem(localId, {
      status: 'error',
      error: getErrorMessage(error, '上传失败')
    })
  }
}

const handleBeforeUpload = (file) => {
  void uploadFile(file)
  return false
}

const getParseMethodOptions = (item) => {
  return (item.parseMethods || []).map((method) => ({
    label: methodLabels[method] || method,
    value: method
  }))
}

const clearParsedState = {
  parsedObjectName: null,
  parsedMinioUrl: null,
  truncated: false,
  parseMethod: null
}

const handleParseMethodChange = (localId, selectedParseMethod) => {
  const item = fileItems.value.find((entry) => entry.localId === localId)
  updateItem(localId, {
    ...clearParsedState,
    selectedParseMethod,
    parseError: null,
    status: item?.status === 'parsed' ? 'uploaded' : item?.status
  })
}

const handleParse = async (item) => {
  if (!item.objectName || !item.selectedParseMethod) return

  updateItem(item.localId, {
    ...clearParsedState,
    status: 'parsing',
    parseError: null
  })
  try {
    const response = await threadApi.parseTmpAttachment({
      object_name: item.objectName,
      file_name: item.fileName,
      bucket_name: item.bucketName,
      parse_method: item.selectedParseMethod
    })
    updateItem(item.localId, {
      status: 'parsed',
      parsedObjectName: response.parsed_object_name,
      parsedMinioUrl: response.parsed_minio_url,
      truncated: response.truncated,
      parseMethod: response.parse_method
    })
    message.success('附件解析完成')
  } catch (error) {
    updateItem(item.localId, {
      ...clearParsedState,
      status: 'uploaded',
      parseError: getErrorMessage(error, '解析失败')
    })
  }
}

const removeItem = (localId) => {
  fileItems.value = fileItems.value.filter((item) => item.localId !== localId)
}

const handleConfirm = async () => {
  if (confirmDisabled.value) return

  const attachments = confirmableItems.value.map((item) => ({
    file_name: item.fileName,
    file_type: item.fileType,
    bucket_name: item.bucketName,
    object_name: item.objectName,
    parsed_object_name: item.parsedObjectName || null,
    truncated: Boolean(item.truncated)
  }))

  confirming.value = true
  try {
    const threadId = props.threadId || (props.ensureThread ? await props.ensureThread() : '')
    if (!threadId) {
      message.error('创建对话失败，无法添加附件')
      return
    }

    const response = await threadApi.confirmTmpThreadAttachments(threadId, attachments)
    message.success('附件已添加')
    emit('added', response)
    emit('update:open', false)
  } catch (error) {
    message.error(getErrorMessage(error, '添加附件失败'))
  } finally {
    confirming.value = false
  }
}

const handleCancel = () => {
  emit('update:open', false)
}

const getStatusColor = (status) => {
  const colorMap = {
    uploading: 'processing',
    uploaded: 'blue',
    parsing: 'processing',
    parsed: 'green',
    error: 'red'
  }
  return colorMap[status] || 'default'
}

const getStatusLabel = (status) => {
  const labelMap = {
    uploading: '上传中',
    uploaded: '已上传',
    parsing: '解析中',
    parsed: '已解析',
    error: '失败'
  }
  return labelMap[status] || status
}

const formatFileSize = (size) => {
  if (!Number.isFinite(size)) return '未知大小'
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}
</script>

<style lang="less" scoped>
.attachment-dropzone {
  margin-bottom: 12px;
}

.dropzone-title {
  margin: 0 0 4px;
  color: var(--gray-800);
  font-size: 14px;
  font-weight: 600;
}

.dropzone-desc {
  margin: 0;
  color: var(--gray-500);
  font-size: 12px;
}

.attachment-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 360px;
  overflow: auto;
}

.attachment-item {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 10px;
  border: 1px solid var(--gray-100);
  border-radius: 8px;
  background: var(--gray-50);
}

.attachment-item-main {
  min-width: 0;
  flex: 1;
}

.attachment-name-row {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.attachment-name {
  overflow: hidden;
  color: var(--gray-900);
  font-size: 13px;
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.attachment-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 4px;
  color: var(--gray-500);
  font-size: 12px;
}

.attachment-error {
  color: var(--color-error-700);
}

.attachment-parsed {
  color: var(--color-success-700);
}

.attachment-actions {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.parse-select {
  width: 150px;
}
</style>
