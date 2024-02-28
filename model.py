import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from config import config

class NCA(tf.keras.Model):
    def __init__(self, channel_n=16, fire_rate=0.5, hidden_size=128, input_channels=3, drop_out_rate=0.25, img_size=28):
        super(NCA, self).__init__()

        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.hidden_size = hidden_size
        self.input_channels = input_channels
        self.drop_out_rate = drop_out_rate
        self.img_size = img_size

        self.p0 = tf.keras.layers.Conv2D(channel_n, kernel_size=3, strides=1, padding='same', activation=None, use_bias=True)
        self.p1 = tf.keras.layers.Conv2D(channel_n, kernel_size=3, strides=1, padding='same', activation=None, use_bias=True)
        self.fc0 = tf.keras.layers.Dense(hidden_size)
        self.fc1 = tf.keras.layers.Dense(channel_n, use_bias=False)
        self.drop0 = tf.keras.layers.Dropout(drop_out_rate)
        self.norm0 = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])

    def perceive(self, x):
        y1 = self.p0(x)
        y2 = self.p1(x)
        y = tf.concat([x, y1, y2], axis=-1)
        
        return y

    def update(self, x):
        
        dx = self.perceive(x)

        dx = self.fc0(dx)
        dx = tf.nn.leaky_relu(dx)

        dx = self.norm0(dx)
        dx = self.drop0(dx)

        dx = self.fc1(dx)

        stochastic = tf.random.uniform((tf.shape(dx)[0], tf.shape(dx)[1], tf.shape(dx)[2],1))
        stochastic = stochastic > self.fire_rate
        stochastic = tf.cast(stochastic, tf.float32)
        dx = dx * stochastic
        x = x+dx
 
        return x

    def call(self, x, steps=10):
        ## batch, height, width, channel
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        x = self.seed(x)
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        for step in range(steps):
            x = self.update(x)
        return x
    
    def seed(self, x):
        seed = tf.zeros((tf.shape(x)[0], self.channel_n-tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]), dtype=tf.float32)
        seed = tf.concat([x, seed], axis=1)
        return seed


class LocalGlobalMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, **kwargs):
        super(LocalGlobalMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.local_mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.global_mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

    def call(self, inputs):
        local_embeddings, global_embeddings = inputs
        
        # Prepare queries, keys, and values for local and global embeddings
        local_queries = tf.keras.layers.Dense(self.d_model)(local_embeddings)
        local_keys = tf.keras.layers.Dense(self.d_model)(local_embeddings)
        local_values = tf.keras.layers.Dense(self.d_model)(local_embeddings)
        
        global_queries = tf.keras.layers.Dense(self.d_model)(global_embeddings)
        global_keys = tf.keras.layers.Dense(self.d_model)(global_embeddings)
        global_values = tf.keras.layers.Dense(self.d_model)(global_embeddings)
        
        # Compute attention for local embeddings
        local_attention = self.local_mha(query=local_queries, key=local_keys, value=local_values)
        
        # Compute attention for global embeddings
        global_attention = self.global_mha(query=global_queries, key=global_keys, value=global_values)
        
        # Combine local and global attention outputs
        # print("local attentoin, embeddings: ", local_attention.shape, local_embeddings.shape)
        # print("global attentoin, embeddings: ", global_attention.shape, global_embeddings.shape)

        local_attention_output = local_attention * local_embeddings
        global_attention_output = global_attention * global_embeddings
        
        return local_attention_output, global_attention_output

class SelfAttention1d(tf.keras.layers.Layer):
    def __init__(self, num_heads, embedding_dim):
        super(SelfAttention1d, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.multi_head_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)

    def call(self, inputs):
        # Reshape the input to (batch_size, seq_len, embedding_dim)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        reshaped_input = tf.reshape(inputs, [batch_size, seq_len, self.embedding_dim])

        # Apply multi-head self-attention
        attention_output = self.multi_head_attn(reshaped_input, reshaped_input)

        # Reshape back to original shape
        output = tf.reshape(attention_output, [batch_size, seq_len])
        output = inputs+output

        return output

class ImageAttention(tf.keras.layers.Layer):
    def __init__(self, d_model=32, num_heads=4, dropout=0.1):
        super(ImageAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.multihead_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, local_features, global_features):
        combined_features = tf.concat([local_features, global_features], axis=-1)
        combined_features2 = tf.reshape(combined_features, (tf.shape(combined_features)[0], -1, self.d_model))
        
        attn_output = self.multihead_attn(combined_features2, combined_features2, combined_features2)
        attn_output = tf.reshape(attn_output, (tf.shape(attn_output)[0], 6, 6, -1))
        # print("fts:", combined_features.shape, attn_output.shape)
        combined_features += attn_output
        combined_features = self.layer_norm(combined_features)
        
        return combined_features


class GLAMORNet(tf.keras.Model):
    def __init__(self, num_classes=7, face_input_shape=(96, 96, 3), context_input_shape=(112, 112, 3)):
        super(GLAMORNet, self).__init__()
        # self.FaceEncodingNet = EncodingNet(input_shape=face_input_shape,
        #                                    num_blocks=config.face_encoding.num_blocks,
        #                                    num_filters=config.face_encoding.num_filters,
        #                                    pooling=config.face_encoding.pooling)

        # self.ContextEncodingNet = EncodingNet(input_shape=context_input_shape,
        #                                       num_blocks=config.context_encoding.num_blocks,
        #                                       num_filters=config.context_encoding.num_filters,
        #                                       pooling=config.context_encoding.pooling)

        self.NCA_face = NCA()
        self.NCA_context = NCA()

        self.NCA_face2 = NCA(channel_n=32)
        self.NCA_context2 = NCA(channel_n=32)

        self.face_attention = SelfAttention1d(num_heads=4, embedding_dim=1)

        self.nca_maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.nca_maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')

        self.self_attention_layer_lg = LocalGlobalMultiHeadAttention(num_heads=4, d_model=32)

        self.GLA = ImageAttention(d_model=64, num_heads=8)

        self.GLAReduction = tf.keras.layers.GlobalAveragePooling2D()  # convert the encoded face tensor to a single vector
        self.face_landmark_fc = tf.keras.layers.Dense(units=128, activation='relu')
        self.head_pose_fc = tf.keras.layers.Dense(units=128, activation='relu')
        
        # GLA module
        self.attention_fc1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.attention_fc1_bn = tf.keras.layers.BatchNormalization()
        self.attention_fc2 = tf.keras.layers.Dense(units=1, activation=None)
        self.attention_dot = tf.keras.layers.Dot(axes=1)

        # Fusion module
        self.face_weight1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.face_weight2 = tf.keras.layers.Dense(units=1, activation=None)
        self.context_weight1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.context_weight2 = tf.keras.layers.Dense(units=1, activation=None)
        self.face_landmark_weight1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.face_landmark_weight2 = tf.keras.layers.Dense(units=1, activation=None)
        self.head_pose_weight1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.head_pose_weight2 = tf.keras.layers.Dense(units=1, activation=None)
        
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)
        self.softmax1 = tf.keras.layers.Activation('softmax')

        # Classifier after fusion
        self.final_fc1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.final_dropout1 = tf.keras.layers.Dropout(rate=config.dropout_rate)
        self.final_classify = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, x_face, x_context, face_data_vector, training=False):
        # face_landmark_vector = face_data_vector[:,:-6]  # (4, 43) 
        # head_pose_vector = face_data_vector[:,-6:]    #(4, 6)

        face_landmark_vector = self.face_attention(face_data_vector)

        # print(face_landmark_vector.shape, head_pose_vector.shape)

        # face = self.FaceEncodingNet(x_face, training=training)  # Get face encoding volume with shape (W,H,C)
        # context = self.ContextEncodingNet(x_context, training=training)
        # face_vector = self.FaceReduction(face)  # dim [1xC]

        nca_x_face = self.NCA_face(x_face)
        nca_x_context = self.NCA_context(x_context)
        nca_x_face = self.nca_maxpool1(nca_x_face)
        nca_x_context = self.nca_maxpool1(nca_x_context)
        nca_x_face = self.NCA_face2(nca_x_face) # 24,24,32
        nca_x_context = self.NCA_context2(nca_x_context) # 28,28,32
        nca_x_face = self.nca_maxpool1(nca_x_face) # 6,6,32
        nca_x_context = self.nca_maxpool1(nca_x_context) # 7,7,32
        nca_x_context = self.nca_maxpool2(nca_x_context) # 6,6,32

        # print("shapes: ", nca_x_context.shape, nca_x_face.shape)


        ## apply self attention to both
        face_attention_output, context_attention_output = self.self_attention_layer_lg([nca_x_face, nca_x_context])
        
        # GLA module
        gla_output = self.GLA(face_attention_output, context_attention_output)
        context_vector = self.GLAReduction(gla_output)

        # GLA module
        # N, H, W, C = context.shape
        # face_vector_repeat = tf.keras.layers.RepeatVector(H * W)(
        #     face_vector)  # clone the vector to W*H vectors shape (H*W,C)
        # context_vector = tf.keras.layers.Reshape((H * W, C))(context)  # tensor with shape (H*W, C)
        # concat1 = tf.keras.layers.Concatenate(axis=-1)([face_vector_repeat,
        #                                                 context_vector])  # concat face vector with each of context location vector to learn attention weight per location
        # attention_weight = self.attention_fc1(concat1)
        # attention_weight = self.attention_fc1_bn(attention_weight, training=training)
        # attention_weight = tf.keras.layers.Activation("relu")(attention_weight)
        # attention_weight = self.attention_fc2(attention_weight)
        # attention_weight = tf.nn.softmax(attention_weight, axis=1)  # a tensor with shape (H*W, 1)
        # context_vector = self.attention_dot(
        #     [context_vector, attention_weight])  # context vector shape (H*W,C) dot with alpha shape (H*W,1) => (1,C)
        # context_vector = tf.keras.layers.Reshape((C,))(
        #     context_vector)  # final context representation (output of the GLA module)

        # Process face landmark and head pose vectors
        face_landmark_vector = self.face_landmark_fc(face_landmark_vector)
        # head_pose_vector = self.head_pose_fc(head_pose_vector)


        # w_f = self.face_weight1(face_vector)
        # w_f = self.face_weight2(w_f)
        w_c = self.context_weight1(context_vector)
        w_c = self.context_weight2(w_c)
        w_fl = self.face_landmark_weight1(face_landmark_vector)
        w_fl = self.face_landmark_weight2(w_fl)
        # w_hp = self.head_pose_weight1(head_pose_vector)
        # w_hp = self.head_pose_weight2(w_hp)        

        w_fclp = self.concat1([w_c, w_fl])
        w_fclp = self.softmax1(w_fclp)

        # face_vector = face_vector * tf.expand_dims(w_fclp[:, 0], axis=-1)
        context_vector = context_vector * tf.expand_dims(w_fclp[:, 0], axis=-1)
        face_landmark_vector =  face_landmark_vector * tf.expand_dims(w_fclp[:, 1], axis=-1)
        # head_pose_vector =  head_pose_vector * tf.expand_dims(w_fclp[:, 2], axis=-1)

        # concat2 = context_vector
        concat2 = tf.keras.layers.Concatenate(axis=-1)([context_vector,face_landmark_vector])
        features = self.final_fc1(concat2)
        features = self.final_dropout1(features, training=training)
        scores = self.final_classify(features)

        return scores


def get_model():
    model = GLAMORNet(config.num_classes, config.face_input_size + [3], config.context_input_size + [3])
    #model.call(tf.keras.Input(config.face_input_size + [3]), tf.keras.Input(config.context_input_size + [3])) #build the model
    return model

if __name__ == '__main__':
    a = tf.random.normal([3, 96, 96, 3])
    b = tf.random.normal([3, 112, 112, 3])
    c = tf.random.normal([3,202])
    print(config.train_images)
    model = get_model()


    model.call(tf.keras.Input((96, 96, 3)), tf.keras.Input((112, 112, 3)),tf.keras.Input((202)))
    o = model(a, b,c, True)
    print(model.summary())
    print(o.shape)



# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf
# from config import config


# class EncodingBlock(tf.keras.Model):
#     def __init__(self, num_filters, input_shape=None, is_pool=True, is_relu=True, conv_filter_size=3, strides=1,
#                  padding='same'):
#         super(EncodingBlock, self).__init__()
#         if (input_shape != None):
#             self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=conv_filter_size,
#                                                input_shape=input_shape, padding=padding, strides=strides)
#         else:
#             self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=conv_filter_size, padding=padding,
#                                                strides=strides)
#         self.bn = tf.keras.layers.BatchNormalization(axis=-1)
#         self.relu = tf.keras.layers.ReLU() if is_relu else None
#         self.pool = tf.keras.layers.MaxPool2D((2, 2)) if is_pool else None

#     def call(self, x, training=False):
#         x = self.conv(x)
#         x = self.bn(x, training)
#         x = self.relu(x) if self.relu != None else x
#         return self.pool(x) if self.pool != None else x


# class EncodingNet(tf.keras.Model):
#     def __init__(self, input_shape, num_blocks, num_filters, pooling=-1):
#         super(EncodingNet, self).__init__()
#         if (pooling == -1):
#             pooling = [True for i in range(num_blocks)]
#         self.blocks = [0 for i in range(num_blocks)]
#         self.blocks[0] = EncodingBlock(num_filters[0], input_shape=input_shape, is_pool=pooling[0])
#         for i in range(1, num_blocks):
#             self.blocks[i] = EncodingBlock(num_filters[i], is_pool=pooling[i])

#     def call(self, x, training=False):
#         for i in range(len(self.blocks)):
#             x = self.blocks[i](x, training)
#         return x


# class GLAMORNet(tf.keras.Model):
#     def __init__(self, num_classes=7, face_input_shape=(96, 96, 3), context_input_shape=(112, 112, 3)):
#         super(GLAMORNet, self).__init__()
#         self.FaceEncodingNet = EncodingNet(input_shape=face_input_shape,
#                                            num_blocks=config.face_encoding.num_blocks,
#                                            num_filters=config.face_encoding.num_filters,
#                                            pooling=config.face_encoding.pooling)

#         self.ContextEncodingNet = EncodingNet(input_shape=context_input_shape,
#                                               num_blocks=config.context_encoding.num_blocks,
#                                               num_filters=config.context_encoding.num_filters,
#                                               pooling=config.context_encoding.pooling)

#         self.FaceReduction = tf.keras.layers.GlobalAveragePooling2D()  # convert the encoded face tensor to a single vector
#         self.face_landmark_fc = tf.keras.layers.Dense(units=128, activation='relu')
#         self.head_pose_fc = tf.keras.layers.Dense(units=128, activation='relu')
        
#         # GLA module
#         self.attention_fc1 = tf.keras.layers.Dense(units=128, activation='relu')
#         self.attention_fc1_bn = tf.keras.layers.BatchNormalization()
#         self.attention_fc2 = tf.keras.layers.Dense(units=1, activation=None)
#         self.attention_dot = tf.keras.layers.Dot(axes=1)

#         # Fusion module
#         self.face_weight1 = tf.keras.layers.Dense(units=128, activation='relu')
#         self.face_weight2 = tf.keras.layers.Dense(units=1, activation=None)
#         self.context_weight1 = tf.keras.layers.Dense(units=128, activation='relu')
#         self.context_weight2 = tf.keras.layers.Dense(units=1, activation=None)
#         self.face_landmark_weight1 = tf.keras.layers.Dense(units=128, activation='relu')
#         self.face_landmark_weight2 = tf.keras.layers.Dense(units=1, activation=None)
#         self.head_pose_weight1 = tf.keras.layers.Dense(units=128, activation='relu')
#         self.head_pose_weight2 = tf.keras.layers.Dense(units=1, activation=None)
        
#         self.concat1 = tf.keras.layers.Concatenate(axis=-1)
#         self.softmax1 = tf.keras.layers.Activation('softmax')

#         # Classifier after fusion
#         self.final_fc1 = tf.keras.layers.Dense(units=128, activation='relu')
#         self.final_dropout1 = tf.keras.layers.Dropout(rate=config.dropout_rate)
#         self.final_classify = tf.keras.layers.Dense(units=num_classes, activation='softmax')

#     def call(self, x_face, x_context,face_data_vector, training=False):
#         face_landmark_vector = face_data_vector[:,:-6] 
#         head_pose_vector = face_data_vector[:,-6:]
#         face = self.FaceEncodingNet(x_face, training=training)  # Get face encoding volume with shape (W,H,C)
#         context = self.ContextEncodingNet(x_context, training=training)
#         face_vector = self.FaceReduction(face)  # dim [1xC]

#         # GLA module
#         N, H, W, C = context.shape
#         face_vector_repeat = tf.keras.layers.RepeatVector(H * W)(
#             face_vector)  # clone the vector to W*H vectors shape (H*W,C)
#         context_vector = tf.keras.layers.Reshape((H * W, C))(context)  # tensor with shape (H*W, C)
#         concat1 = tf.keras.layers.Concatenate(axis=-1)([face_vector_repeat,
#                                                         context_vector])  # concat face vector with each of context location vector to learn attention weight per location
#         attention_weight = self.attention_fc1(concat1)
#         attention_weight = self.attention_fc1_bn(attention_weight, training=training)
#         attention_weight = tf.keras.layers.Activation("relu")(attention_weight)
#         attention_weight = self.attention_fc2(attention_weight)
#         attention_weight = tf.nn.softmax(attention_weight, axis=1)  # a tensor with shape (H*W, 1)
#         context_vector = self.attention_dot(
#             [context_vector, attention_weight])  # context vector shape (H*W,C) dot with alpha shape (H*W,1) => (1,C)
#         context_vector = tf.keras.layers.Reshape((C,))(
#             context_vector)  # final context representation (output of the GLA module)

#         # Process face landmark and head pose vectors
#         face_landmark_vector = self.face_landmark_fc(face_landmark_vector)
#         head_pose_vector = self.head_pose_fc(head_pose_vector)


#         w_f = self.face_weight1(face_vector)
#         w_f = self.face_weight2(w_f)
#         w_c = self.context_weight1(context_vector)
#         w_c = self.context_weight2(w_c)
#         w_fl = self.face_landmark_weight1(face_landmark_vector)
#         w_fl = self.face_landmark_weight2(w_fl)
#         w_hp = self.head_pose_weight1(head_pose_vector)
#         w_hp = self.head_pose_weight2(w_hp)        

#         w_fclp = self.concat1([w_f, w_c,w_fl,w_hp])
#         w_fclp = self.softmax1(w_fclp)

#         face_vector = face_vector * tf.expand_dims(w_fclp[:, 0], axis=-1)
#         context_vector = context_vector * tf.expand_dims(w_fclp[:, 1], axis=-1)
#         face_landmark_vector =  face_landmark_vector * tf.expand_dims(w_fclp[:, 2], axis=-1)
#         head_pose_vector =  head_pose_vector * tf.expand_dims(w_fclp[:, 3], axis=-1)

#         # concat2 = context_vector
#         concat2 = tf.keras.layers.Concatenate(axis=-1)([face_vector, context_vector,face_landmark_vector,head_pose_vector])
#         features = self.final_fc1(concat2)
#         features = self.final_dropout1(features, training=training)
#         scores = self.final_classify(features)

#         return scores


# def get_model():
#     model = GLAMORNet(config.num_classes, config.face_input_size + [3], config.context_input_size + [3])
#     #model.call(tf.keras.Input(config.face_input_size + [3]), tf.keras.Input(config.context_input_size + [3])) #build the model
#     return model

# if __name__ == '__main__':
#     a = tf.random.normal([3, 96, 96, 3])
#     b = tf.random.normal([3, 112, 112, 3])
#     c = tf.random.normal([3,202])
#     print(config.train_images)
#     model = get_model()


#     model.call(tf.keras.Input((96, 96, 3)), tf.keras.Input((112, 112, 3)),tf.keras.Input((202)))
#     o = model(a, b,c, True)
#     print(model.summary())
#     print(o.shape)
