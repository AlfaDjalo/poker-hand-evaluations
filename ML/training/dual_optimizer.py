import tensorflow as tf

class DualOptimizerModel(tf.keras.Model):
    """
    Wrapper that uses separate optimizers for encoder and head layers.
    
    Usage:
        base_model = build_model(config, mode_config)
        model = DualOptimizerModel(
            base_model,
            encoder_lr=1e-4,
            head_lr=1e-3
        )
        model.compile(loss=loss, loss_weights=loss_weights)
    """
    
    def __init__(self, base_model, encoder_lr=1e-4, head_lr=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self._encoder_lr = encoder_lr
        self._head_lr = head_lr
                
        # Create two optimizers
        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=encoder_lr)
        self.head_optimizer = tf.keras.optimizers.Adam(learning_rate=head_lr)
        
        # Split variables by layer type
        self._split_variables()
    
    def get_config(self):
        """Serialization config for saving/loading."""
        config = super().get_config()
        config.update({
            "encoder_lr": self._encoder_lr,
            "head_lr": self._head_lr,
        })
        return config
    
    def _split_variables(self):
        """Split trainable variables into encoder and head groups."""
        from ML.models import CardStateEncoder, CardSetEncoder
        
        self.encoder_vars = []
        self.head_vars = []
        
        for layer in self.base_model.layers:
            # Check if layer is encoder type
            if isinstance(layer, (CardStateEncoder, CardSetEncoder)):
                self.encoder_vars.extend(layer.trainable_variables)
            else:
                # Check nested layers (for heads containing encoders)
                if hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        if isinstance(sublayer, (CardStateEncoder, CardSetEncoder)):
                            self.encoder_vars.extend(sublayer.trainable_variables)
                        else:
                            self.head_vars.extend(sublayer.trainable_variables)
                else:
                    self.head_vars.extend(layer.trainable_variables)
        
        print(f"\n🔧 Variable Split:")
        print(f"  Encoder variables: {len(self.encoder_vars)}")
        print(f"  Head variables: {len(self.head_vars)}")
    
    def call(self, inputs, training=False):
        """Forward pass through base model."""
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        """Custom training step with dual optimizers."""
        x, y = data
        
        # Use persistent=True to compute gradients multiple times
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            y_pred = self(x, training=True)
            
            # Compute loss
            loss = self.compute_loss(x, y, y_pred)
        
        # Compute gradients for encoder and head separately
        if self.encoder_vars:
            encoder_grads = tape.gradient(loss, self.encoder_vars)
            self.encoder_optimizer.apply_gradients(zip(encoder_grads, self.encoder_vars))
        
        if self.head_vars:
            head_grads = tape.gradient(loss, self.head_vars)
            self.head_optimizer.apply_gradients(zip(head_grads, self.head_vars))
        
        # Delete the tape to free resources
        del tape
        
        # Update metrics
        self.compute_metrics(x, y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """Validation step (no optimization)."""
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        self.compute_metrics(x, y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}