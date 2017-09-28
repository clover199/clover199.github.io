<?php
function message($msg = '', $type = 'success')
{
    if (!empty($msg)) {
        $output = '<p class="' . $type . '">' . $msg . '</p>';
    } else {
        $output = '';
    }
    return $output;
}
function display_commentbox($num = 5)
{
    $db = DB::getInstance();
    /* DISPLAY SHOUTS*/
    $sql = 'SELECT * FROM comment ORDER BY id DESC LIMIT '. $num .';';
    $query = $db->query($sql);
    $query->setFetchMode(PDO::FETCH_OBJ);
    ob_start();
?>
<ul class="commentbox">

<?php
while ($comment = $query->fetch()) {
    // create our Gravatar from each users email address.
    $grav_url = "http://www.gravatar.com/avatar.php?gravatar_id=" . md5( strtolower($comment->email) ) . "&size=70";
?>
    <li class="comment">
        <h4 class="title"> <?php echo $comment->name; ?> at <?php echo $comment->created_at; ?> </h4>
        <div class="avatar">
            <img src="<?php echo $grav_url; ?>" alt="Gravatar" />
        </div>
        <p>
            <?php echo $comment->comment; ?>
        </p>
    </li>
<?php
}
?>
</ul>
<h2 style="text-align:left;">Shout!</h2>
<form class="commentform" action="" method="post">
    <div>
        <label for="name">Name: </label>
        <input id="name" name="name" type="text" />
    </div>
    <div>
        <label for="email">Email: </label>
        <input id="email" name="email" type="email" />
    </div>
    <div>
        <label for="comment">Comment:</label>
        <textarea id="comment" name="comment" rows="5"></textarea>
    </div>
    <div>
        <input type="submit" name="submit" value="Send" />
    </div>
</form>

<?php
    return ob_get_contents();
}
