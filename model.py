import tensorflow as tf


regularization_strength = 0.01

class Attention(tf.keras.layers.Layer):
    def __init__(self, in_dim_k, in_dim_q, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        
        self.num_heads = num_heads
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define the layers
        self.q = tf.keras.layers.Dense(out_dim, use_bias=qkv_bias)
        self.kv = tf.keras.layers.Dense(out_dim * 2, use_bias=qkv_bias)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(out_dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)
        self.qkmatrix = None
        # self.post_attention_pooling = tf.keras.layers.MaxPooling1D(pool_size=2) 

    def call(self, inputs, training=False):
        x, x_q = inputs
        _, Nk, Ck = x.shape
        _, Nq, Cq = x_q.shape

        q = self.q(x_q)
        q = tf.reshape(q, [-1, Nq, self.num_heads, tf.shape(q)[-1]])
        q = tf.transpose(q, [0, 2, 1, 3])

        kv = self.kv(x)
        kv = tf.reshape(kv, [-1, Nk, 2, self.num_heads, tf.shape(kv)[-1] // 2])
        kv = tf.transpose(kv, [2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        
        self.qkmatrix = attn
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, Nq, self.num_heads * tf.shape(x)[-1]])
        x = self.proj(x)
        x = self.proj_drop(x)
        # x = self.post_attention_pooling(x)

        return x, self.qkmatrix


# Video stages
def video_stage1(input_shape):
    video_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(video_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    shape = x.get_shape().as_list()
    x = tf.keras.layers.Reshape((shape[1]*shape[2], shape[3]))(x)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
    # x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return tf.keras.Model(inputs=video_input, outputs=x, name="video_stage1_model")

def video_stage2(input_shape_stage2):
    # Stage 2
    vedio_input_stage2 = tf.keras.layers.Input(shape=input_shape_stage2)
    x2 = tf.keras.layers.Conv1D(128, 3, activation='relu')(vedio_input_stage2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    # x2 = tf.keras.layers.MaxPooling1D(pool_size=2)(x2)
    x2 = tf.keras.layers.Conv1D(256, 3, activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling1D(pool_size=2)(x2)
    x2 = tf.keras.layers.Dropout(0.3)(x2)
    return tf.keras.Model(inputs=vedio_input_stage2, outputs=x2, name="video_stage2_model")

def audio_stage1(input_shape):
    audio_input = tf.keras.layers.Input(shape=input_shape)
    # x = tf.keras.layers.Reshape((-1, 1))(audio_input)
    x = tf.keras.layers.Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu')(audio_input)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, padding='same')(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return tf.keras.Model(inputs=audio_input, outputs=x, name="audio_stage1_model")

def audio_stage2(input_shape_stage2):
    audio_input_stage2 = tf.keras.layers.Input(shape=input_shape_stage2)
    x2 = tf.keras.layers.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu')(audio_input_stage2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    # x2 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, padding='same')(x2)
    x2 = tf.keras.layers.Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, padding='same')(x2)
    x2 = tf.keras.layers.Dropout(0.3)(x2)
    return tf.keras.Model(inputs=audio_input_stage2, outputs=x2, name="audio_stage2_model")



class Model(tf.keras.Model):
    def __init__(self, video_input_shape, audio_input_shape, num_heads=1):
        super(Model, self).__init__()
        self.video_stage1 = video_stage1(video_input_shape)
        self.audio_stage1 = audio_stage1(audio_input_shape)



        # Getting the shapes after stage 1 transformations
        video_output_shape = self.video_stage1.layers[-1].output_shape[1:]
        audio_output_shape = self.audio_stage1.layers[-1].output_shape[1:]


        audio_output_shape_after_attention = (audio_output_shape[0], 64)
        vedio_output_shape_after_attention = (video_output_shape[0], 64)
        # audio_output_shape_after_attention = (video_output_shape[-1])
        # vedio_output_shape_after_attention = (audio_output_shape[-1])

        self.audio_stage2 = audio_stage2(audio_output_shape_after_attention)
        self.video_stage2 = video_stage2(vedio_output_shape_after_attention)

        self.av1 = Attention(in_dim_k=video_output_shape[-1], in_dim_q=audio_output_shape[-1], out_dim=audio_output_shape[-1], num_heads=num_heads)
        self.va1 = Attention(in_dim_k=audio_output_shape[-1], in_dim_q=video_output_shape[-1], out_dim=video_output_shape[-1], num_heads=num_heads)

        self.final_mlp_layer_1 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_strength))
        self.final_mlp_layer_output = tf.keras.layers.Dense(8, activation='softmax', name="final_classification")
        self.batch_normalization = tf.keras.layers.BatchNormalization()

    def call(self,inputs, training=False):
        x_audio, x_visual = inputs
        x_audio = self.audio_stage1(x_audio)
        x_visual = self.video_stage1(x_visual)

        proj_x_a = tf.transpose(x_audio, perm=[0,2,1])
        proj_x_v = tf.transpose(x_visual, perm=[0,2,1])

        _, h_av = self.av1((proj_x_v, proj_x_a))
        _, h_va = self.va1((proj_x_a, proj_x_v))

        if h_av.shape[1] > 1:  # if more than 1 head, take average
            h_av = tf.reduce_mean(h_av, axis=1, keepdims=True)

        if h_va.shape[1] > 1:  # if more than 1 head, take average
            h_va = tf.reduce_mean(h_va, axis=1, keepdims=True)

        h_av = tf.reduce_sum(h_av, axis=-2)
        h_va = tf.reduce_sum(h_va, axis=-2)

        x_audio = h_va + x_audio
        x_visual = h_av + x_visual
        
        # Passing through stage 2
        x_audio = self.audio_stage2(x_audio)
        x_visual = self.video_stage2(x_visual)
        flat_audio = tf.keras.layers.Flatten()(x_audio)
        flat_video = tf.keras.layers.Flatten()(x_visual)

        combined = tf.keras.layers.concatenate([flat_audio, flat_video])
        hidden = self.final_mlp_layer_1(combined)
        hidden = self.batch_normalization(hidden)
        hidden = tf.keras.layers.Dropout(0.3)(hidden)
        sentiment = self.final_mlp_layer_output(hidden)
        
        return sentiment



class Predictor():
    def __init__(self):
        super(Predictor, self).__init__()
        self.model = Model(video_input_shape=(100, 48, 48), audio_input_shape=(162,1))
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.00001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        # Create dummy data
        dummy_audio_data = tf.random.normal([1, *self.model.audio_stage1.input_shape[1:]])
        dummy_video_data = tf.random.normal([1, *self.model.video_stage1.input_shape[1:]])

        # Pass the dummy data through the model
        _ = self.model([dummy_audio_data, dummy_video_data])

        # Now you can view the summary
        self.model.summary()

        self.model.load_weights("my_model_weights.h5")

    def predict(self,inputData):
        return self.model.predict(inputData)
        